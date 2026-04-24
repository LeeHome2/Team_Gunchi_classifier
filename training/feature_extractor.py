"""
엔티티 → 모델 입력 feature 벡터.

학습/추론 공용. joblib으로 pipeline 저장해서 불러쓴다.

Feature 구성:
 - TF-IDF on raw_layer (char n-gram 2~4) — 다국어/표기변이 흡수
 - Entity type one-hot
 - 지오메트리: log1p(length), log1p(bbox_w/h), aspect_ratio
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GEOMETRIC_ENTITY_TYPES  # noqa: E402

# 고정된 entity type 어휘 (학습/추론 일관성)
ENTITY_TYPE_VOCAB = sorted(
    GEOMETRIC_ENTITY_TYPES
    | {"TEXT", "MTEXT", "DIMENSION", "HATCH", "INSERT", "3DFACE", "SOLID", "LEADER", "POINT"}
)


GEOMETRY_COLUMNS = ["length", "bbox_width", "bbox_height", "aspect_ratio"]


class FeatureExtractor:
    """엔티티 DataFrame/dict 리스트 → 2D numpy 배열."""

    def __init__(
        self,
        tfidf_min_df: int = 2,
        tfidf_max_features: int = 500,
        ngram_range: tuple[int, int] = (2, 4),
    ):
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=tfidf_min_df,
            max_features=tfidf_max_features,
            lowercase=True,
        )
        self.entity_types: List[str] = ENTITY_TYPE_VOCAB
        self.fitted = False
        self.feature_dim: int = 0

    # ─── fit / transform ────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":
        layers = df["raw_layer"].fillna("").astype(str).tolist()
        # 빈 문자열 방지 (TfidfVectorizer가 vocabulary 비면 에러)
        if all(len(s) == 0 for s in layers):
            layers = ["unknown"]
        self.tfidf.fit(layers)
        self.fitted = True
        # feature_dim 사전 계산
        self.feature_dim = (
            len(self.tfidf.vocabulary_) + len(self.entity_types) + len(GEOMETRY_COLUMNS)
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeatureExtractor.fit()를 먼저 호출해야 합니다.")

        n = len(df)

        # (1) TF-IDF on raw_layer
        layers = df["raw_layer"].fillna("").astype(str).tolist()
        tfidf_vec = self.tfidf.transform(layers).toarray()

        # (2) Entity type one-hot
        type_onehot = np.zeros((n, len(self.entity_types)), dtype=np.float32)
        for i, etype in enumerate(df["entity_type"].fillna("")):
            if etype in self.entity_types:
                type_onehot[i, self.entity_types.index(etype)] = 1.0

        # (3) 지오메트리 (log1p로 스케일 압축)
        # copy=True 필수: pandas가 read-only view 반환할 수 있어서 inplace 할당 실패 방지
        geom = df[GEOMETRY_COLUMNS].fillna(0).astype(np.float64).to_numpy(copy=True)
        # length, bbox_width, bbox_height는 log1p (절대값 사용), aspect_ratio는 그대로
        geom[:, 0:3] = np.log1p(np.abs(geom[:, 0:3]))
        # aspect_ratio 이상치 clip (0~10)
        geom[:, 3] = np.clip(geom[:, 3], 0, 10)

        return np.concatenate([tfidf_vec, type_onehot, geom], axis=1).astype(np.float32)

    # ─── 편의 메서드 ────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def transform_entities(self, entities: List[dict]) -> np.ndarray:
        """추론 시 엔티티 dict 리스트 → feature 배열."""
        df = pd.DataFrame(entities)
        # parse_dxf에 있는 컬럼들이 없을 수도 있으니 채움
        for col in ["raw_layer", "entity_type", *GEOMETRY_COLUMNS]:
            if col not in df.columns:
                df[col] = None
        return self.transform(df)

    # ─── persistence ────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "tfidf": self.tfidf,
            "entity_types": self.entity_types,
            "fitted": self.fitted,
            "feature_dim": self.feature_dim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureExtractor":
        obj = cls()
        obj.tfidf = d["tfidf"]
        obj.entity_types = d["entity_types"]
        obj.fitted = d["fitted"]
        obj.feature_dim = d.get("feature_dim", 0)
        return obj
