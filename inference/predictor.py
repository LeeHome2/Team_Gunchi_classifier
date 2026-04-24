"""
모델 로드 + 예측.

상주 서빙에서 사용. CPU 기반 (학과 서버 GPU 독점 금지 규정 준수).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES, MODEL_DIR  # noqa: E402
from training.feature_extractor import FeatureExtractor  # noqa: E402


class ModelBundle:
    """로드된 모델 + feature pipeline + 메타."""

    def __init__(self, run_id: str, model, extractor: FeatureExtractor, config: dict, metrics: dict):
        self.run_id = run_id
        self.model = model
        self.extractor = extractor
        self.config = config
        self.metrics = metrics

    def predict(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """엔티티 dict 리스트 → 예측 결과."""
        if not entities:
            return []

        X = self.extractor.transform_entities(entities)
        preds = self.model.predict(X)
        probas = self.model.predict_proba(X)

        # sklearn 예측 클래스 순서에 맞게 confidence 매핑
        class_idx = {c: i for i, c in enumerate(self.model.classes_)}

        results = []
        for i, ent in enumerate(entities):
            pred_class = str(preds[i])
            conf = float(probas[i, class_idx[pred_class]])
            results.append({
                "entity_id": ent.get("entity_id"),
                "raw_layer": ent.get("raw_layer"),
                "entity_type": ent.get("entity_type"),
                "predicted_class": pred_class,
                "confidence": round(conf, 4),
            })
        return results


# ─── 모듈 전역 캐시 (FastAPI 라이프사이클) ─────────────────────
_bundle: Optional[ModelBundle] = None
_load_lock = RLock()


def load_bundle(run_id: str) -> ModelBundle:
    """디스크에서 모델 로드."""
    out_dir = MODEL_DIR / run_id
    if not out_dir.exists():
        raise FileNotFoundError(f"모델 폴더 없음: {out_dir}")

    model = joblib.load(out_dir / "model.joblib")
    extractor_state = joblib.load(out_dir / "feature_pipeline.joblib")
    extractor = FeatureExtractor.from_dict(extractor_state)

    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    metrics = {}
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    return ModelBundle(run_id, model, extractor, config, metrics)


def get_active_bundle() -> ModelBundle:
    """현재 active 모델 번들 반환. 없으면 레지스트리에서 조회 후 로드."""
    global _bundle
    if _bundle is not None:
        return _bundle

    with _load_lock:
        if _bundle is not None:
            return _bundle
        from mlops.registry import get_active
        active = get_active()
        if not active:
            raise RuntimeError(
                "활성 모델이 없습니다. 학습 후 mlops/registry.py:set_active(run_id) 호출 필요"
            )
        _bundle = load_bundle(active["run_id"])
        return _bundle


def reload_active_bundle() -> ModelBundle:
    """active 모델 재로드 (deploy API 호출 후)."""
    global _bundle
    with _load_lock:
        _bundle = None
    return get_active_bundle()


def classify_entities(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """엔티티 리스트 → 분류 결과 요약."""
    bundle = get_active_bundle()
    predictions = bundle.predict(entities)

    # 요약
    from collections import Counter
    class_counts: Counter = Counter(p["predicted_class"] for p in predictions)
    confidences = [p["confidence"] for p in predictions]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    # 레이어별 집계 (wall로 분류된 레이어명 선정 — building_cesium 매스 생성에 사용)
    from collections import defaultdict
    layer_class_votes: Dict[str, Counter] = defaultdict(Counter)
    for p in predictions:
        layer = p.get("raw_layer") or ""
        layer_class_votes[layer][p["predicted_class"]] += 1

    layer_decisions: Dict[str, str] = {}
    for layer, votes in layer_class_votes.items():
        layer_decisions[layer] = votes.most_common(1)[0][0]

    return {
        "model_version": bundle.run_id,
        "total_entities": len(entities),
        "class_counts": dict(class_counts),
        "average_confidence": round(avg_conf, 4),
        "predictions": predictions,
        "layer_decisions": layer_decisions,
        "layers": sorted(layer_decisions.keys()),
    }
