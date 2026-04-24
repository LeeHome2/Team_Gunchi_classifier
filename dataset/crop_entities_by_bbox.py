"""
정규화 bbox(0~1) × CAD extents = CAD 좌표 bbox.
엔티티 CSV에서 bbox 안의 엔티티만 남긴 별도 CSV 생성.

다중 평면도(1F/2F 등)면 평면도별로 개별 CSV 출력.

핵심 주의사항:
 - 이미지 좌표는 좌상단이 (0,0), 우하단이 (1,1)
 - CAD 좌표는 좌하단이 최소값, 우상단이 최대값
 - → y축 반전 필요

사용법:
    # 단일 file_id 처리
    python -m dataset.crop_entities_by_bbox --file-id foo

    # 배치: processed/ 폴더의 모든 CSV + 대응 bboxes.json
    python -m dataset.crop_entities_by_bbox --processed-dir data/processed \
        --preview-dir data/preview \
        --output-dir data/cropped
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PREVIEW_DIR, PROCESSED_DIR  # noqa: E402


def norm_bbox_to_cad(
    norm_bbox: Dict[str, float], extents_cad: Dict[str, float]
) -> Dict[str, float]:
    """정규화 bbox(0~1) → CAD 좌표계 bbox. y축 반전."""
    x_min_cad = extents_cad["min_x"]
    y_min_cad = extents_cad["min_y"]
    width_cad = extents_cad["max_x"] - extents_cad["min_x"]
    height_cad = extents_cad["max_y"] - extents_cad["min_y"]

    return {
        "x_min": x_min_cad + norm_bbox["x_min"] * width_cad,
        "x_max": x_min_cad + norm_bbox["x_max"] * width_cad,
        # y축 반전: image y=0은 CAD max_y, image y=1은 CAD min_y
        "y_max": extents_cad["max_y"] - norm_bbox["y_min"] * height_cad,
        "y_min": extents_cad["max_y"] - norm_bbox["y_max"] * height_cad,
    }


def _entity_intersects_bbox(row: pd.Series, cad_bbox: Dict[str, float]) -> bool:
    """엔티티 bbox가 cad_bbox와 교차하는지 (느슨: AABB intersect)."""
    if pd.isna(row.get("bbox_x_min")):
        return False
    try:
        return not (
            row["bbox_x_max"] < cad_bbox["x_min"]
            or row["bbox_x_min"] > cad_bbox["x_max"]
            or row["bbox_y_max"] < cad_bbox["y_min"]
            or row["bbox_y_min"] > cad_bbox["y_max"]
        )
    except (TypeError, KeyError):
        return False


def crop_file(
    file_id: str,
    csv_path: Path,
    preview_meta_path: Path,
    bboxes_path: Path,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """
    1개 DXF 파일에 대해 평면도별 크롭 CSV 생성.

    Returns: [{"label": "...", "path": "...", "n_entities": N, "cad_bbox": {...}}, ...]
    """
    df = pd.read_csv(csv_path)
    preview = json.loads(preview_meta_path.read_text(encoding="utf-8"))
    bboxes_data = json.loads(bboxes_path.read_text(encoding="utf-8"))

    extents_cad = preview.get("extents_cad")
    if not extents_cad:
        raise ValueError(f"{preview_meta_path.name}에 extents_cad가 없음")

    floorplans = bboxes_data.get("floorplans", [])
    if not bboxes_data.get("floorplans_found") or not floorplans:
        # Fallback: vLLM이 평면도 못 찾았으면 전체 이미지를 평면도로 간주
        # (데이터셋이 거의 다 평면도이므로 이 가정이 대체로 맞음)
        floorplans = [{
            "label": "full",
            "reason": "fallback: vLLM이 평면도 검출 실패 → 전체 이미지 사용",
            "bbox": {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
        }]

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Dict[str, Any]] = []

    for fp in floorplans:
        label = fp.get("label", "fp")
        norm_bbox = fp.get("bbox")
        if not norm_bbox:
            continue

        cad_bbox = norm_bbox_to_cad(norm_bbox, extents_cad)

        # 엔티티 필터
        mask = df.apply(lambda r: _entity_intersects_bbox(r, cad_bbox), axis=1)
        cropped = df[mask].copy()
        cropped["floorplan_label"] = label
        # file_id 보존 (학습 시 파일 단위 split 유지)
        if "file_id" not in cropped.columns:
            cropped["file_id"] = file_id

        # 안전한 파일명 (label에 공백/특수문자 있을 수 있음)
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        out_path = output_dir / f"{file_id}__{safe_label}.csv"
        cropped.to_csv(out_path, index=False)

        outputs.append({
            "label": label,
            "path": str(out_path),
            "n_entities": int(len(cropped)),
            "cad_bbox": cad_bbox,
            "norm_bbox": norm_bbox,
        })

    return outputs


def crop_batch(
    processed_dir: Path,
    preview_dir: Path,
    output_dir: Path,
    bboxes_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """processed_dir의 모든 *.csv에 대해 대응 preview/bboxes 찾아 크롭."""
    bboxes_dir = bboxes_dir or processed_dir

    csvs = sorted(processed_dir.glob("*.csv"))
    ok = skip = fail = 0
    total_cropped = 0

    for csv_path in csvs:
        file_id = csv_path.stem
        preview_meta = preview_dir / f"{file_id}.preview.json"
        bboxes_path = bboxes_dir / f"{file_id}.bboxes.json"

        if not preview_meta.exists():
            print(f"SKIP {file_id}: preview 메타 없음")
            skip += 1
            continue
        if not bboxes_path.exists():
            print(f"SKIP {file_id}: bboxes 없음")
            skip += 1
            continue

        try:
            outputs = crop_file(
                file_id, csv_path, preview_meta, bboxes_path, output_dir
            )
            if not outputs:
                print(f"--  {file_id}: 평면도 0개 (vLLM이 검출 실패)")
                skip += 1
                continue
            n_out = len(outputs)
            n_ents = sum(o["n_entities"] for o in outputs)
            total_cropped += n_ents
            print(f"OK  {file_id}: {n_out}개 평면도, {n_ents} entities")
            ok += 1
        except Exception as e:
            print(f"ERR {file_id}: {type(e).__name__}: {e}")
            fail += 1

    return {"ok": ok, "skip": skip, "fail": fail, "total_cropped_entities": total_cropped}


def main() -> None:
    ap = argparse.ArgumentParser(description="bbox로 엔티티 CSV 크롭")
    ap.add_argument("--file-id", help="단일 file_id 처리")
    ap.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--preview-dir", default=str(PREVIEW_DIR))
    ap.add_argument("--bboxes-dir", default=None, help="bboxes.json 위치 (기본: processed-dir)")
    ap.add_argument(
        "--output-dir",
        default=str(PROCESSED_DIR.parent / "cropped"),
        help="출력 폴더 (기본: data/cropped)",
    )
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    preview_dir = Path(args.preview_dir)
    output_dir = Path(args.output_dir)
    bboxes_dir = Path(args.bboxes_dir) if args.bboxes_dir else processed_dir

    if args.file_id:
        csv_path = processed_dir / f"{args.file_id}.csv"
        preview_meta = preview_dir / f"{args.file_id}.preview.json"
        bboxes_path = bboxes_dir / f"{args.file_id}.bboxes.json"
        outputs = crop_file(args.file_id, csv_path, preview_meta, bboxes_path, output_dir)
        for o in outputs:
            print(f"  {o['label']}: {o['n_entities']} entities → {Path(o['path']).name}")
        print(f"\n완료: {len(outputs)}개 평면도")
    else:
        result = crop_batch(processed_dir, preview_dir, output_dir, bboxes_dir)
        print(f"\n전체 결과: OK={result['ok']} SKIP={result['skip']} FAIL={result['fail']}")
        print(f"크롭된 총 엔티티 수: {result['total_cropped_entities']}")


if __name__ == "__main__":
    main()
