"""
학습 스크립트. 로컬에서 1회성 실행.

입력:  data/labeled/*.csv (weak_label 컬럼 포함)
출력:  models/saved/{run_id}/
         ├─ model.joblib
         ├─ feature_pipeline.joblib
         ├─ config.json     (하이퍼파라미터 + 메타)
         └─ metrics.json    (val/test 성능)
       mlops.db 에 experiment row 기록

사용법:
    python -m training.train                          # 기본 설정
    python -m training.train --run-id v1_test         # run_id 지정
    python -m training.train --input-dir data/labeled --max-iter 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES, MODEL_DIR, TRAIN_PARAMS_PATH  # noqa: E402
from training.feature_extractor import FeatureExtractor  # noqa: E402

# XGBoost 래퍼 클래스 (모듈 레벨에 정의해야 pickle 가능)
try:
    import xgboost as xgb
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.preprocessing import LabelEncoder

    class XGBClassifierWrapper(BaseEstimator, ClassifierMixin):
        """XGBoost를 sklearn 호환되게 감싸는 래퍼 (문자열 라벨 지원)."""
        _estimator_type = "classifier"

        def __init__(self, n_estimators=400, max_depth=7, learning_rate=0.08,
                     random_state=42, tree_method="hist", eval_metric="mlogloss", **kwargs):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.learning_rate = learning_rate
            self.random_state = random_state
            self.tree_method = tree_method
            self.eval_metric = eval_metric
            self.kwargs = kwargs
            self.classes_ = None
            self.label_encoder = None
            self.xgb_model = None

        def fit(self, X, y):
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            self.classes_ = self.label_encoder.classes_
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                tree_method=self.tree_method,
                eval_metric=self.eval_metric,
                **self.kwargs,
            )
            self.xgb_model.fit(X, y_encoded)
            return self

        def predict(self, X):
            y_pred_encoded = self.xgb_model.predict(X)
            return self.label_encoder.inverse_transform(y_pred_encoded)

        def predict_proba(self, X):
            return self.xgb_model.predict_proba(X)

        def get_params(self, deep=True):
            return {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "random_state": self.random_state,
                "tree_method": self.tree_method,
                "eval_metric": self.eval_metric,
                **self.kwargs,
            }

        def set_params(self, **params):
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    self.kwargs[key] = value
            return self

    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


def write_progress(run_id: str, progress: int, message: str) -> None:
    """진행률 파일에 현재 상태 기록 (API에서 폴링)."""
    progress_file = MODEL_DIR / "saved" / f"{run_id}.progress"
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    data = {"progress": progress, "message": message}
    progress_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_labeled_data(input_dir: Path) -> pd.DataFrame:
    """data/labeled/*.csv 전부 읽어서 하나의 DataFrame으로."""
    csvs = sorted(input_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"라벨된 CSV가 없습니다: {input_dir}")

    dfs = []
    for c in csvs:
        try:
            df = pd.read_csv(c)
            if "weak_label" not in df.columns:
                print(f"[warn] {c.name}에 weak_label 컬럼 없음 → 스킵")
                continue
            dfs.append(df)
        except Exception as e:
            print(f"[warn] {c.name} 읽기 실패: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"로드: {len(csvs)}개 파일 → {len(combined)} rows")
    return combined


def split_by_file(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """파일 단위 train/val/test split. 같은 파일 row가 train+test에 동시 들어가는 걸 방지."""
    rng = np.random.default_rng(seed)
    files = df["file_id"].unique()
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = set(files[:n_train])
    val_files = set(files[n_train : n_train + n_val])
    test_files = set(files[n_train + n_val :])

    return {
        "train": df[df["file_id"].isin(train_files)].reset_index(drop=True),
        "val": df[df["file_id"].isin(val_files)].reset_index(drop=True),
        "test": df[df["file_id"].isin(test_files)].reset_index(drop=True),
        "train_files": sorted(train_files),
        "val_files": sorted(val_files),
        "test_files": sorted(test_files),
    }


def evaluate_split(
    model,
    extractor: FeatureExtractor,
    df: pd.DataFrame,
    split_name: str,
) -> Dict:
    X = extractor.transform(df)
    y_true = df["weak_label"].tolist()
    y_pred = model.predict(X).tolist()
    y_proba = model.predict_proba(X)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()
    report = classification_report(
        y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
    )

    # confidence 통계
    confidences = y_proba.max(axis=1)
    return {
        "split": split_name,
        "n": len(df),
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "per_class": {
            c: {
                "precision": round(report[c]["precision"], 4),
                "recall": round(report[c]["recall"], 4),
                "f1": round(report[c]["f1-score"], 4),
                "support": int(report[c]["support"]),
            }
            for c in CLASSES if c in report
        },
        "confusion_matrix_labels": CLASSES,
        "confusion_matrix": cm,
        "confidence_mean": round(float(confidences.mean()), 4),
        "confidence_low_ratio": round(float((confidences < 0.6).mean()), 4),
    }


def train_main(
    input_dir: Path,
    run_id: str,
    max_iter: int,
    max_depth: int,
    learning_rate: float,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    model_type: str = "hist_gradient",
) -> Dict:
    print("=" * 60)
    print(f"학습 시작: run_id={run_id}")
    print(f"분할 비율: train={train_ratio:.2f} / val={val_ratio:.2f} / test={1-train_ratio-val_ratio:.2f}")
    print("=" * 60)
    write_progress(run_id, 0, "학습 초기화 중...")

    # 비율 유효성 검증
    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError(
            f"잘못된 분할 비율: train={train_ratio}, val={val_ratio} "
            f"(0<train<1, 0<=val<1, train+val<1 이어야 함)"
        )

    # 1. 데이터 로드
    write_progress(run_id, 5, "데이터 로딩 중...")
    df = load_labeled_data(input_dir)
    write_progress(run_id, 15, "데이터 로딩 완료")
    print(f"전체 라벨 분포:\n{df['weak_label'].value_counts().to_dict()}\n")

    # 2. 파일 단위 split
    write_progress(run_id, 20, "데이터 분할 중...")
    splits = split_by_file(df, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    print(f"Split — train: {len(splits['train'])} / val: {len(splits['val'])} / test: {len(splits['test'])}")
    print(f"  파일 수: train={len(splits['train_files'])}, val={len(splits['val_files'])}, test={len(splits['test_files'])}\n")

    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    # 3. Feature extractor fit on train
    write_progress(run_id, 25, "특성 추출기 학습 중...")
    print("Feature extractor fit...")
    extractor = FeatureExtractor()
    extractor.fit(train_df)
    print(f"  feature_dim = {extractor.feature_dim}")

    X_train = extractor.transform(train_df)
    X_val = extractor.transform(val_df) if len(val_df) > 0 else None
    X_test = extractor.transform(test_df) if len(test_df) > 0 else None

    y_train = train_df["weak_label"].tolist()

    # 4. 학습 - model_type에 따라 분기
    write_progress(run_id, 35, "모델 학습 시작...")
    t0 = time.time()

    if model_type == "hist_gradient":
        print(f"HistGradientBoostingClassifier 학습 (max_iter={max_iter}, max_depth={max_depth})...")
        model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            early_stopping=True if X_val is not None and len(val_df) > 50 else False,
            validation_fraction=None,
        )
    elif model_type == "xgboost":
        if not _XGBOOST_AVAILABLE:
            raise ImportError("xgboost가 설치되지 않았습니다. pip install xgboost")
        print(f"XGBClassifier 학습 (n_estimators={max_iter}, max_depth={max_depth})...")
        model = XGBClassifierWrapper(
            n_estimators=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            tree_method="hist",
            eval_metric="mlogloss",
        )
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        print(f"RandomForestClassifier 학습 (n_estimators={max_iter}, max_depth={max_depth})...")
        model = RandomForestClassifier(
            n_estimators=max_iter,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

    model_type_name = type(model).__name__
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  학습 완료: {train_time:.1f}s")
    write_progress(run_id, 60, "모델 학습 완료")

    # 5. 확률 보정 (Probability Calibration)
    # 트리 기반 모델은 과신(overconfident) 경향이 있어 calibration 필요
    # XGBoost는 자체 확률 보정이 양호하고 sklearn CalibratedClassifierCV 호환 이슈로 skip
    if model_type == "xgboost":
        print("  (XGBoost — calibration skip, 자체 확률 보정 사용)\n")
    elif X_val is not None and len(val_df) >= 30:
        write_progress(run_id, 70, "확률 보정 적용 중...")
        print("  확률 보정(isotonic) 적용 중...")
        calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calibrated_model.fit(X_val, val_df["weak_label"].tolist())
        model = calibrated_model
        print("  확률 보정 완료\n")
    else:
        print("  (검증셋 부족으로 확률 보정 생략)\n")

    # 6. 평가
    write_progress(run_id, 80, "모델 평가 시작...")
    metrics_all = {
        "train": evaluate_split(model, extractor, train_df, "train"),
    }
    if X_val is not None and len(val_df) > 0:
        metrics_all["val"] = evaluate_split(model, extractor, val_df, "val")
    if X_test is not None and len(test_df) > 0:
        metrics_all["test"] = evaluate_split(model, extractor, test_df, "test")

    for split_name, m in metrics_all.items():
        print(f"[{split_name:<5s}] n={m['n']}  acc={m['accuracy']:.3f}  f1_macro={m['f1_macro']:.3f}  "
              f"f1_weighted={m['f1_weighted']:.3f}  conf_mean={m['confidence_mean']:.3f}")
    write_progress(run_id, 90, "모델 평가 완료")

    # 7. 저장
    write_progress(run_id, 95, "모델 저장 중...")
    out_dir = MODEL_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(extractor.to_dict(), out_dir / "feature_pipeline.joblib")

    is_calibrated = isinstance(model, CalibratedClassifierCV)
    config = {
        "run_id": run_id,
        "model_type": model_type_name,
        "calibrated": is_calibrated,
        "sklearn_based": model_type != "xgboost",
        "hyperparams": {
            "max_iter": max_iter,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": round(1 - train_ratio - val_ratio, 4),
        },
        "features": {
            "feature_dim": extractor.feature_dim,
            "tfidf_vocab_size": len(extractor.tfidf.vocabulary_),
            "entity_type_vocab_size": len(extractor.entity_types),
            "geometry_features": ["length", "bbox_width", "bbox_height", "aspect_ratio"],
        },
        "classes": CLASSES,
        "train_info": {
            "train_files": splits["train_files"],
            "val_files": splits["val_files"],
            "test_files": splits["test_files"],
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "training_time_seconds": round(train_time, 2),
        },
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_all, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n모델 저장: {out_dir}")
    write_progress(run_id, 100, "학습 완료!")
    return {"run_id": run_id, "out_dir": str(out_dir), "metrics": metrics_all, "config": config}


def main() -> None:
    # 기본 하이퍼파라미터 로드
    default_params = {}
    if TRAIN_PARAMS_PATH.exists():
        try:
            default_params = json.loads(TRAIN_PARAMS_PATH.read_text())
        except Exception:
            default_params = {}

    ap = argparse.ArgumentParser(description="레이어 분류기 학습")
    ap.add_argument(
        "--input-dir",
        default="data/labeled",
        help="weak_label 포함 CSV 디렉토리",
    )
    ap.add_argument("--run-id", default=None, help="실험 ID (기본: 자동 생성)")
    ap.add_argument("--max-iter", type=int, default=default_params.get("n_estimators", 400))
    ap.add_argument("--max-depth", type=int, default=default_params.get("max_depth", 7))
    ap.add_argument("--learning-rate", type=float, default=default_params.get("learning_rate", 0.08))
    ap.add_argument("--seed", type=int, default=default_params.get("random_seed", 42))
    ap.add_argument(
        "--train-ratio", type=float,
        default=default_params.get("train_ratio", 0.70),
        help="훈련 데이터 비율 (기본 0.70)",
    )
    ap.add_argument(
        "--val-ratio", type=float,
        default=default_params.get("val_ratio", 0.15),
        help="검증 데이터 비율 (기본 0.15). test_ratio = 1 - train - val",
    )
    ap.add_argument("--no-mlops-log", action="store_true", help="mlops.db 기록 스킵")
    ap.add_argument(
        "--model-type",
        default="hist_gradient",
        choices=["hist_gradient", "random_forest", "xgboost"],
        help="모델 타입 (기본: hist_gradient)",
    )
    args = ap.parse_args()

    run_id = args.run_id or f"v_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    input_dir = Path(args.input_dir)

    result = train_main(
        input_dir=input_dir,
        run_id=run_id,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        model_type=args.model_type,
    )

    # MLOps DB 기록
    if not args.no_mlops_log:
        try:
            from mlops.registry import record_experiment
            record_experiment(
                run_id=run_id,
                model_type=result["config"]["model_type"],
                hyperparams=result["config"]["hyperparams"],
                model_path=result["out_dir"],
                metrics=result["metrics"],
                train_info=result["config"]["train_info"],
            )
            print(f"\nMLOps DB 기록 완료: run_id={run_id}")
        except Exception as e:
            print(f"[warn] MLOps DB 기록 실패: {e}")


if __name__ == "__main__":
    main()
