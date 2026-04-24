"""
weak label 생성기.

parse_dxf CSV → weak_label 컬럼 추가된 CSV.

4단계 필터:
 1) 노이즈 레이어 제거 (PDF 임포트 잔해, 워터마크 등) → other
 2) 비지오메트리 엔티티 (TEXT, DIMENSION, HATCH, INSERT 등) → other
 3) 키워드 매칭 (wall/door/window)
 4) 지오메트리 sanity check (너무 짧은 선분, 도면 전체를 덮는 테두리)

사용법:
    python -m dataset.weak_label -i data/processed/foo.csv
    python -m dataset.weak_label --input-dir data/processed --output-dir data/labeled
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES, GEOMETRIC_ENTITY_TYPES, PROCESSED_DIR  # noqa: E402


# ─── 1단계: 노이즈 레이어 패턴 ───────────────────────────────
NOISE_PATTERNS = [
    r"^PDF\d*_",            # CAD에 PDF 임포트한 잔해
    r"cadblocks",           # cadblocksfree.com 워터마크
    r"^Defpoints$",         # AutoCAD 내부 표시용
    r"^\$\d+",              # 블록 참조 잔해 (예: $0$xBDAFP01$0$)
]


# ─── 3단계: 4-class 키워드 사전 ──────────────────────────────
# 순서 주의: wall 매칭이 door/window 매칭보다 선행 (벽체가 가장 흔함)
KEYWORDS = {
    "wall": [
        r"\bwall\b",
        r"walls",
        r"muro",
        r"\b壁\b",
        r"\b벽\b",
        r"parede",
    ],
    "door": [
        r"\bdoor\b",
        r"doors",
        r"puerta",
        r"porte",
        r"\b문\b",
        r"tür",
        r"(?:^|[-_\s])d(?:$|[-_\s])",  # "D" 단독 약어
    ],
    "window": [
        r"\bwindow\b",
        r"windows",
        r"ventana",
        r"fenetre",
        r"\bwin\b",
        r"\bwinows\b",   # 오타
        r"\b창\b",
    ],
}


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def weak_label_row(row: pd.Series, doc_extent_max: Optional[float] = None) -> str:
    """엔티티 1개 → 4-class 중 하나."""
    layer = str(row.get("raw_layer") or "")
    etype = str(row.get("entity_type") or "")

    # 1) 노이즈 필터
    if _matches_any(layer, NOISE_PATTERNS):
        return "other"

    # 2) 비지오메트리 엔티티는 자동 other
    if etype not in GEOMETRIC_ENTITY_TYPES:
        return "other"

    # 3) 키워드 매칭 (우선순위: wall → door → window)
    for cls in ("wall", "door", "window"):
        if _matches_any(layer, KEYWORDS[cls]):
            # 4) 지오메트리 sanity check
            # 선분이 도면 전체의 70% 이상이면 테두리로 간주 → other
            if doc_extent_max and row.get("bbox_width") and row.get("bbox_height"):
                bw = float(row["bbox_width"])
                bh = float(row["bbox_height"])
                if max(bw, bh) > doc_extent_max * 0.7:
                    return "other"
            return cls

    return "other"


def label_csv(csv_path: Path, meta_path: Optional[Path] = None) -> pd.DataFrame:
    """parse_dxf CSV + meta.json → weak_label 컬럼 추가된 DataFrame."""
    df = pd.read_csv(csv_path)

    # meta.json 읽어서 도면 크기 추출 (sanity check에 사용)
    doc_extent_max = None
    if meta_path is None:
        meta_path = csv_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            extents = meta.get("extents", {})
            w = extents.get("width") or 0
            h = extents.get("height") or 0
            doc_extent_max = max(float(w), float(h)) if (w and h) else None
        except Exception:
            pass

    df["weak_label"] = df.apply(lambda r: weak_label_row(r, doc_extent_max), axis=1)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Weak label 생성 (키워드 + 지오메트리)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("-i", "--input", help="입력 CSV (parse_dxf 출력)")
    g.add_argument("--input-dir", help="입력 CSV 디렉토리 (배치)")
    ap.add_argument(
        "-o", "--output-dir",
        default=str(PROCESSED_DIR.parent / "labeled"),
        help="출력 폴더 (기본: data/labeled)"
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        paths = [Path(args.input)]
    else:
        paths = sorted(Path(args.input_dir).glob("*.csv"))

    total_ok = 0
    total_dist: dict[str, int] = {c: 0 for c in CLASSES}

    for p in paths:
        try:
            df = label_csv(p)
            out = output_dir / p.name
            df.to_csv(out, index=False)

            dist = df["weak_label"].value_counts().to_dict()
            for c in CLASSES:
                total_dist[c] += dist.get(c, 0)

            line = f"OK  {p.name}: n={len(df)}"
            for c in CLASSES:
                line += f" {c}={dist.get(c, 0)}"
            print(line)
            total_ok += 1
        except Exception as e:
            print(f"ERR {p.name}: {e}")

    print(f"\n완료: {total_ok}개 파일")
    print(f"전체 라벨 분포:")
    total = sum(total_dist.values())
    for c in CLASSES:
        pct = (total_dist[c] / total * 100) if total > 0 else 0
        print(f"  {c:<7s} {total_dist[c]:>8} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
