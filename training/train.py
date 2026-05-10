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
) -> Dict:
    print("=" * 60)
    print(f"학습 시작: run_id={run_id}")
    print(f"분할 비율: train={train_ratio:.2f} / val={val_ratio:.2f} / test={1-train_ratio-val_ratio:.2f}")
    print("=" * 60)

    # 비율 유효성 검증
    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError(
            f"잘못된 분할 비율: train={train_ratio}, val={val_ratio} "
            f"(0<train<1, 0<=val<1, train+val<1 이어야 함)"
        )

    # 1. 데이터 로드
    df = load_labeled_data(input_dir)
    print(f"전체 라벨 분포:\n{df['weak_label'].value_counts().to_dict()}\n")

    # 2. 파일 단위 split
    splits = split_by_file(df, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    print(f"Split — train: {len(splits['train'])} / val: {len(splits['val'])} / test: {len(splits['test'])}")
    print(f"  파일 수: train={len(splits['train_files'])}, val={len(splits['val_files'])}, test={len(splits['test_files'])}\n")

    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    # 3. Feature extractor fit on train
    print("Feature extractor fit...")
    extractor = FeatureExtractor()
    extractor.fit(train_df)
    print(f"  feature_dim = {extractor.feature_dim}")

    X_train = extractor.transform(train_df)
    X_val = extractor.transform(val_df) if len(val_df) > 0 else None
    X_test = extractor.transform(test_df) if len(test_df) > 0 else None

    y_train = train_df["weak_label"].tolist()

    # 4. 학습
    print(f"HistGradientBoosting 학습 (max_iter={max_iter}, max_depth={max_depth})...")
    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        early_stopping=True if X_val is not None and len(val_df) > 50 else False,
        validation_fraction=None,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  학습 완료: {train_time:.1f}s\n")

    # 5. 평가
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

    # 6. 저장
    out_dir = MODEL_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(extractor.to_dict(), out_dir / "feature_pipeline.joblib")

    config = {
        "run_id": run_id,
        "model_type": "HistGradientBoostingClassifier",
        "sklearn_based": True,
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
