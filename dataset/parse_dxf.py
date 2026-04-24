"""
DXF 파일 → 엔티티별 CSV + 파일 메타 JSON.

모든 후속 단계(크롭, 라벨링, 학습, 추론)의 공통 입력 포맷.
한 행 = 한 엔티티. 지오메트리 feature까지 같이 뽑는다.

사용법:
    # 단일 파일
    python -m dataset.parse_dxf -i ../데이터셋1-dxf/dxf/sample.dxf

    # 배치
    python -m dataset.parse_dxf --input-dir ../데이터셋1-dxf/dxf
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import ezdxf
from ezdxf import bbox as ezbbox

# 프로젝트 루트를 import path에 추가 (단독 실행 대비)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GEOMETRIC_ENTITY_TYPES, PROCESSED_DIR  # noqa: E402


# ─── CSV 스키마 ────────────────────────────────────────────
CSV_COLUMNS = [
    "file_id",
    "entity_id",
    "entity_type",
    "raw_layer",
    # 지오메트리 공통
    "length",
    "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max",
    "bbox_width", "bbox_height", "aspect_ratio",
    "center_x", "center_y",
    # 타입별 속성
    "start_x", "start_y", "end_x", "end_y",      # LINE
    "radius", "start_angle", "end_angle",         # CIRCLE / ARC
    "n_vertices", "is_closed",                    # LWPOLYLINE / POLYLINE
    "text_content",                               # TEXT / MTEXT (앞 100자)
    # 파생
    "is_geometric",                               # True면 wall/door/window 후보
]


# ─── 지오메트리 계산 ───────────────────────────────────────
def _entity_length(e) -> Optional[float]:
    """엔티티의 총 길이. 계산 불가하면 None."""
    t = e.dxftype()
    try:
        if t == "LINE":
            s, p = e.dxf.start, e.dxf.end
            return math.hypot(p.x - s.x, p.y - s.y)

        if t == "LWPOLYLINE":
            pts = list(e.get_points("xy"))
            if len(pts) < 2:
                return 0.0
            total = sum(
                math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                for i in range(len(pts) - 1)
            )
            if e.closed and len(pts) > 2:
                total += math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])
            return total

        if t == "POLYLINE":
            verts = [v.dxf.location for v in e.vertices]
            if len(verts) < 2:
                return 0.0
            return sum(
                math.hypot(verts[i + 1].x - verts[i].x, verts[i + 1].y - verts[i].y)
                for i in range(len(verts) - 1)
            )

        if t == "ARC":
            r = e.dxf.radius
            sa, ea = e.dxf.start_angle, e.dxf.end_angle
            sweep = (ea - sa) % 360.0
            return r * math.radians(sweep)

        if t == "CIRCLE":
            return 2 * math.pi * e.dxf.radius

        if t == "ELLIPSE":
            # Ramanujan 근사
            major = e.dxf.major_axis.magnitude
            minor = major * e.dxf.ratio
            h = ((major - minor) / (major + minor)) ** 2 if (major + minor) > 0 else 0
            return math.pi * (major + minor) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
    except Exception:
        return None
    return None


def _entity_bbox(e) -> Optional[Dict[str, float]]:
    """엔티티의 2D bounding box. 계산 불가하면 None."""
    try:
        bb = ezbbox.extents([e], fast=True)
        if bb is None or not bb.has_data:
            return None
        return {
            "x_min": float(bb.extmin.x),
            "y_min": float(bb.extmin.y),
            "x_max": float(bb.extmax.x),
            "y_max": float(bb.extmax.y),
        }
    except Exception:
        return None


def _guess_unit(width: Optional[float], height: Optional[float]) -> str:
    """도면 전체 크기로 단위 추정. 주택 평면도는 보통 10~50m 범위."""
    if not width or not height:
        return "unknown"
    m = max(width, height)
    if m > 1000:
        return "mm"
    if m > 100:
        return "cm"
    return "m"


# ─── 파싱 본체 ─────────────────────────────────────────────
def parse_dxf(dxf_path: Path) -> Dict[str, Any]:
    """DXF 1개 파싱 → {entities: [...], meta: {...}}."""
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    file_id = dxf_path.stem

    # 전체 extents
    try:
        ex = ezbbox.extents(msp, fast=True)
        if ex and ex.has_data:
            ext_min_x, ext_min_y = float(ex.extmin.x), float(ex.extmin.y)
            ext_max_x, ext_max_y = float(ex.extmax.x), float(ex.extmax.y)
        else:
            ext_min_x = ext_min_y = ext_max_x = ext_max_y = None
    except Exception:
        ext_min_x = ext_min_y = ext_max_x = ext_max_y = None

    ext_w = (ext_max_x - ext_min_x) if ext_min_x is not None else None
    ext_h = (ext_max_y - ext_min_y) if ext_min_y is not None else None
    unit = _guess_unit(ext_w, ext_h)

    entities: List[Dict[str, Any]] = []
    layer_counts: Counter = Counter()

    for idx, e in enumerate(msp):
        t = e.dxftype()
        layer = e.dxf.layer if hasattr(e.dxf, "layer") else "0"
        layer_counts[layer] += 1

        row: Dict[str, Any] = {col: None for col in CSV_COLUMNS}
        row["file_id"] = file_id
        row["entity_id"] = f"{file_id}__{idx}"
        row["entity_type"] = t
        row["raw_layer"] = layer
        row["is_geometric"] = t in GEOMETRIC_ENTITY_TYPES

        # 공통 지오메트리
        row["length"] = _entity_length(e)
        bb = _entity_bbox(e)
        if bb:
            row["bbox_x_min"] = bb["x_min"]
            row["bbox_y_min"] = bb["y_min"]
            row["bbox_x_max"] = bb["x_max"]
            row["bbox_y_max"] = bb["y_max"]
            row["bbox_width"] = bb["x_max"] - bb["x_min"]
            row["bbox_height"] = bb["y_max"] - bb["y_min"]
            row["center_x"] = (bb["x_min"] + bb["x_max"]) / 2
            row["center_y"] = (bb["y_min"] + bb["y_max"]) / 2
            if row["bbox_height"] and row["bbox_height"] > 1e-9:
                row["aspect_ratio"] = row["bbox_width"] / row["bbox_height"]

        # 타입별 속성
        try:
            if t == "LINE":
                row["start_x"] = float(e.dxf.start.x)
                row["start_y"] = float(e.dxf.start.y)
                row["end_x"] = float(e.dxf.end.x)
                row["end_y"] = float(e.dxf.end.y)
            elif t == "ARC":
                row["radius"] = float(e.dxf.radius)
                row["start_angle"] = float(e.dxf.start_angle)
                row["end_angle"] = float(e.dxf.end_angle)
            elif t == "CIRCLE":
                row["radius"] = float(e.dxf.radius)
            elif t == "LWPOLYLINE":
                pts = list(e.get_points("xy"))
                row["n_vertices"] = len(pts)
                row["is_closed"] = bool(e.closed)
            elif t == "POLYLINE":
                verts = list(e.vertices)
                row["n_vertices"] = len(verts)
                row["is_closed"] = bool(getattr(e, "is_closed", False))
            elif t in ("TEXT", "MTEXT"):
                try:
                    if hasattr(e, "plain_text"):
                        txt = e.plain_text()
                    else:
                        txt = getattr(e.dxf, "text", "")
                    row["text_content"] = str(txt)[:100]
                except Exception:
                    row["text_content"] = ""
        except Exception:
            # 개별 엔티티 속성 추출 실패는 무시 (다른 row로 계속 진행)
            pass

        entities.append(row)

    meta: Dict[str, Any] = {
        "file_id": file_id,
        "filename": dxf_path.name,
        "dxf_version": doc.dxfversion,
        "total_entities": len(entities),
        "extents": {
            "min_x": ext_min_x,
            "min_y": ext_min_y,
            "max_x": ext_max_x,
            "max_y": ext_max_y,
            "width": ext_w,
            "height": ext_h,
        },
        "estimated_unit": unit,
        "unique_layers": len(layer_counts),
        "layer_entity_counts": dict(layer_counts.most_common()),
    }

    return {"entities": entities, "meta": meta}


# ─── I/O ───────────────────────────────────────────────────
def write_outputs(result: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    """엔티티 CSV + 메타 JSON을 output_dir에 저장."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_id = result["meta"]["file_id"]
    csv_path = output_dir / f"{file_id}.csv"
    meta_path = output_dir / f"{file_id}.meta.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(result["entities"])

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(result["meta"], f, indent=2, ensure_ascii=False, default=str)

    return {"csv": csv_path, "meta": meta_path}


# ─── CLI ───────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="DXF → CSV + 메타 JSON 추출")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("-i", "--input", help="입력 DXF 파일 경로")
    g.add_argument("--input-dir", help="입력 DXF 디렉토리 (배치)")
    ap.add_argument("-o", "--output-dir", default=str(PROCESSED_DIR), help="출력 폴더")
    ap.add_argument("--limit", type=int, help="배치 모드에서 처리 개수 제한 (디버그용)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)

    if args.input:
        paths = [Path(args.input)]
    else:
        paths = sorted(Path(args.input_dir).glob("*.dxf"))
        if args.limit:
            paths = paths[: args.limit]

    ok = fail = 0
    for p in paths:
        try:
            result = parse_dxf(p)
            out = write_outputs(result, output_dir)
            n = result["meta"]["total_entities"]
            u = result["meta"]["estimated_unit"]
            layers = result["meta"]["unique_layers"]
            print(f"OK  {p.name}: {n} entities, {layers} layers, unit={u} -> {out['csv'].name}")
            ok += 1
        except Exception as ex:
            print(f"ERR {p.name}: {ex}")
            fail += 1

    print(f"\n완료: 성공 {ok} / 실패 {fail}")


if __name__ == "__main__":
    main()
