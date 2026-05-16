"""
학습 데이터 빌드 orchestrator.

전체 파이프라인을 하나의 명령으로:
    DXF → parse → render → detect_floorplan → crop → weak_label → labeled CSV

실패한 파일은 스킵하고 계속 진행.
각 단계 결과를 리포트로 출력.

사용법:
    # 전체 (학과 vLLM 실제 호출)
    python -m dataset.build_training_dataset --dxf-dir "../데이터셋1-dxf/dxf"

    # 일부만 (테스트)
    python -m dataset.build_training_dataset --dxf-dir "../데이터셋1-dxf/dxf" --limit 5

    # vLLM 없이 mock 모드 (검증용)
    python -m dataset.build_training_dataset --dxf-dir "../데이터셋1-dxf/dxf" --mock --limit 5

    # 특정 단계만 (기존 결과 재활용)
    python -m dataset.build_training_dataset --skip-parse --skip-render
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PREVIEW_DIR, PROCESSED_DIR, REPORT_DIR  # noqa: E402

# 진행률 파일 경로 (전역 - main에서 설정)
_PROGRESS_FILE: Path | None = None


def write_progress(progress: int, message: str) -> None:
    """진행률 파일에 현재 상태 기록 (API에서 폴링)."""
    if _PROGRESS_FILE is None:
        return
    try:
        data = {"progress": progress, "message": message}
        _PROGRESS_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def stage_parse(dxf_dir: Path, limit: int | None) -> Dict[str, int]:
    from dataset.parse_dxf import parse_dxf, write_outputs

    paths = sorted(dxf_dir.glob("*.dxf"))
    if limit:
        paths = paths[:limit]

    ok = fail = 0
    for p in paths:
        try:
            result = parse_dxf(p)
            write_outputs(result, PROCESSED_DIR)
            ok += 1
        except Exception as e:
            print(f"  [parse ERR] {p.name}: {e}")
            fail += 1
    return {"stage": "parse", "ok": ok, "fail": fail, "total": len(paths)}


def stage_render(limit: int | None) -> Dict[str, int]:
    from dataset.render_preview import render_dxf_to_png

    # processed/에 있는 file_id들에 대해서만 render (parse 성공한 파일만)
    meta_files = sorted(PROCESSED_DIR.glob("*.meta.json"))
    if limit:
        meta_files = meta_files[:limit]

    ok = fail = skip = 0
    from config import EXTERNAL_DATASET_DIR
    for meta in meta_files:
        file_id = meta.stem.replace(".meta", "")
        dxf_path = EXTERNAL_DATASET_DIR / f"{file_id}.dxf"
        png_path = PREVIEW_DIR / f"{file_id}.png"

        if png_path.exists():
            skip += 1
            continue

        if not dxf_path.exists():
            fail += 1
            continue

        try:
            render_dxf_to_png(dxf_path, PREVIEW_DIR)
            ok += 1
        except Exception as e:
            print(f"  [render ERR] {dxf_path.name}: {e}")
            fail += 1

    return {"stage": "render", "ok": ok, "skip_cached": skip, "fail": fail}


def stage_detect(mock: bool, limit: int | None) -> Dict[str, int]:
    from dataset.detect_floorplan import detect_floorplan_for_file

    pngs = sorted(PREVIEW_DIR.glob("*.png"))
    if limit:
        pngs = pngs[:limit]

    ok = fail = cached = 0
    t0 = time.time()
    for png in pngs:
        cache_path = PROCESSED_DIR / f"{png.stem}.bboxes.json"
        was_cached = cache_path.exists()

        try:
            detect_floorplan_for_file(png, cache_dir=PROCESSED_DIR, mock=mock)
            if was_cached:
                cached += 1
            ok += 1
        except Exception as e:
            print(f"  [detect ERR] {png.name}: {type(e).__name__}: {e}")
            fail += 1

    dt = time.time() - t0
    return {"stage": "detect", "ok": ok, "cached": cached, "fail": fail, "elapsed_s": round(dt, 1)}


def stage_crop() -> Dict[str, int]:
    from dataset.crop_entities_by_bbox import crop_batch

    cropped_dir = PROCESSED_DIR.parent / "cropped"
    return {"stage": "crop", **crop_batch(PROCESSED_DIR, PREVIEW_DIR, cropped_dir)}


def stage_label() -> Dict[str, int]:
    from dataset.weak_label import label_csv

    cropped_dir = PROCESSED_DIR.parent / "cropped"
    labeled_dir = PROCESSED_DIR.parent / "labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(cropped_dir.glob("*.csv"))
    ok = fail = 0
    dist = {"wall": 0, "door": 0, "window": 0, "other": 0}

    for csv_path in csvs:
        try:
            # crop된 CSV의 meta는 원본 meta 재사용
            original_file_id = csv_path.stem.split("__")[0]
            meta_path = PROCESSED_DIR / f"{original_file_id}.meta.json"
            df = label_csv(csv_path, meta_path=meta_path)

            for c in dist:
                dist[c] += int((df["weak_label"] == c).sum())

            df.to_csv(labeled_dir / csv_path.name, index=False)
            ok += 1
        except Exception as e:
            print(f"  [label ERR] {csv_path.name}: {e}")
            fail += 1

    return {"stage": "label", "ok": ok, "fail": fail, "distribution": dist}


def main() -> None:
    global _PROGRESS_FILE

    ap = argparse.ArgumentParser(description="학습 데이터 빌드 전체 파이프라인")
    ap.add_argument(
        "--dxf-dir",
        default=None,
        help="원본 DXF 디렉토리 (기본: config.EXTERNAL_DATASET_DIR)",
    )
    ap.add_argument("--limit", type=int, help="파일 개수 제한 (디버그)")
    ap.add_argument("--mock", action="store_true", help="vLLM 호출을 mock으로 대체")
    ap.add_argument("--skip-parse", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--skip-detect", action="store_true")
    ap.add_argument("--skip-crop", action="store_true")
    ap.add_argument("--skip-label", action="store_true")
    ap.add_argument("--job-id", default=None, help="작업 ID (진행률 추적용)")
    args = ap.parse_args()

    # 진행률 파일 초기화
    if args.job_id:
        _PROGRESS_FILE = REPORT_DIR / f"{args.job_id}.progress"
        _PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

    from config import EXTERNAL_DATASET_DIR
    dxf_dir = Path(args.dxf_dir) if args.dxf_dir else EXTERNAL_DATASET_DIR

    print("=" * 60)
    print(f"학습 데이터 빌드 시작")
    print(f"  DXF 폴더: {dxf_dir}")
    print(f"  Mock 모드: {args.mock}")
    print(f"  Limit: {args.limit or '전체'}")
    print("=" * 60)
    write_progress(0, "빌드 초기화 중...")

    report: List[Dict] = []

    if not args.skip_parse:
        print("\n[1/5] parse_dxf")
        write_progress(5, "[1/5] DXF 파싱 중...")
        r = stage_parse(dxf_dir, args.limit)
        report.append(r)
        print(f"  → OK={r['ok']} FAIL={r['fail']}")

    if not args.skip_render:
        print("\n[2/5] render_preview")
        write_progress(20, "[2/5] 미리보기 렌더링 중...")
        r = stage_render(args.limit)
        report.append(r)
        print(f"  → OK={r['ok']} SKIP(cached)={r.get('skip_cached',0)} FAIL={r['fail']}")

    if not args.skip_detect:
        print(f"\n[3/5] detect_floorplan (vLLM Vision {'MOCK' if args.mock else 'REAL'})")
        write_progress(40, "[3/5] 평면도 영역 감지 중... (시간 소요)")
        r = stage_detect(args.mock, args.limit)
        report.append(r)
        print(f"  → OK={r['ok']} CACHED={r.get('cached',0)} FAIL={r['fail']} ({r['elapsed_s']}s)")

    if not args.skip_crop:
        print("\n[4/5] crop_entities_by_bbox")
        write_progress(70, "[4/5] 엔티티 크롭 중...")
        r = stage_crop()
        report.append(r)
        print(f"  → OK={r['ok']} SKIP={r['skip']} FAIL={r['fail']} "
              f"(crop된 엔티티 {r['total_cropped_entities']}개)")

    if not args.skip_label:
        print("\n[5/5] weak_label")
        write_progress(85, "[5/5] 라벨링 중...")
        r = stage_label()
        report.append(r)
        print(f"  → OK={r['ok']} FAIL={r['fail']}")
        print(f"  → 라벨 분포: {r['distribution']}")

    # 최종 리포트 저장
    write_progress(95, "리포트 저장 중...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"build_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(
        json.dumps(
            {"args": vars(args), "stages": report},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n리포트 저장: {report_path}")
    write_progress(100, "빌드 완료!")


if __name__ == "__main__":
    main()
