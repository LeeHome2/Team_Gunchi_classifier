"""
학과 vLLM Vision 모델로 평면도 bbox 검출.

입력: DXF 렌더 PNG (dataset/render_preview.py 결과)
출력: bbox JSON (정규화 좌표 0~1), 파일당 {file_id}.bboxes.json

주요 기능:
 - 파일당 1회만 호출하도록 캐시
 - --mock 모드: vLLM 호출 없이 "전체 이미지 = 1 평면도"로 처리 (개발/테스트)
 - 실패해도 다음 파일 진행

사용법:
    # 단일 파일
    python -m dataset.detect_floorplan -i data/preview/foo.png

    # 배치 (모든 PNG)
    python -m dataset.detect_floorplan --input-dir data/preview

    # Mock 모드 (vLLM 호출 없이)
    python -m dataset.detect_floorplan --input-dir data/preview --mock
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FLOORPLAN_PROMPT_PATH, LLM_VISION_MODEL, PROCESSED_DIR  # noqa: E402


# ─── mock 응답 (vLLM 없이 테스트용) ──────────────────────────
def _mock_response() -> Dict[str, Any]:
    return {
        "floorplans_found": True,
        "floorplans": [
            {
                "label": "fp0",
                "reason": "[MOCK] 전체 이미지를 단일 평면도로 간주",
                "bbox": {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
            }
        ],
    }


def detect_floorplan_for_file(
    png_path: Path,
    cache_dir: Optional[Path] = None,
    model: Optional[str] = None,
    use_cache: bool = True,
    mock: bool = False,
) -> Dict[str, Any]:
    """단일 PNG → bbox dict. 캐시 있으면 재활용."""
    cache_dir = Path(cache_dir or PROCESSED_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{png_path.stem}.bboxes.json"

    if use_cache and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass  # 캐시 파싱 실패 시 무시하고 재호출

    if mock:
        result = _mock_response()
        cache_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result

    # 실제 vLLM 호출
    from llm.client import ask_vision
    from llm.parse_response import extract_json_from_text, validate_floorplan_response

    prompt = FLOORPLAN_PROMPT_PATH.read_text(encoding="utf-8")
    system = "너는 건축 CAD 도면에서 평면도 영역을 찾는 도우미이며, 반드시 JSON으로만 응답해야 한다."

    raw = ask_vision(
        image_path=png_path,
        prompt=prompt,
        system=system,
        model=model or LLM_VISION_MODEL,
        response_format_json=True,
    )
    parsed = extract_json_from_text(raw)
    validated = validate_floorplan_response(parsed)

    cache_path.write_text(
        json.dumps(validated, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return validated


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM Vision → 평면도 bbox 검출")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("-i", "--input", help="입력 PNG 파일")
    g.add_argument("--input-dir", help="입력 PNG 디렉토리")
    ap.add_argument(
        "--cache-dir",
        default=str(PROCESSED_DIR),
        help="bbox JSON 저장/캐시 폴더 (기본: data/processed)",
    )
    ap.add_argument("--no-cache", action="store_true", help="기존 캐시 무시하고 재호출")
    ap.add_argument("--mock", action="store_true", help="vLLM 없이 mock 응답 (전체 이미지 = 1 평면도)")
    ap.add_argument("--limit", type=int, help="배치에서 처리 개수 제한")
    ap.add_argument("--model", default=None, help="모델 별칭 (기본: config.LLM_VISION_MODEL)")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    use_cache = not args.no_cache

    if args.input:
        paths = [Path(args.input)]
    else:
        paths = sorted(Path(args.input_dir).glob("*.png"))
        if args.limit:
            paths = paths[: args.limit]

    ok = fail = cached = 0
    for p in paths:
        cache_path = cache_dir / f"{p.stem}.bboxes.json"
        was_cached = use_cache and cache_path.exists()

        try:
            t0 = time.time()
            result = detect_floorplan_for_file(
                p, cache_dir=cache_dir, model=args.model,
                use_cache=use_cache, mock=args.mock,
            )
            dt = time.time() - t0

            n_fp = len(result.get("floorplans", []))
            mark = "[cache]" if was_cached else ("[mock]" if args.mock else f"[vllm {dt:.1f}s]")
            print(f"OK  {p.name}: floorplans={n_fp} {mark}")
            if was_cached:
                cached += 1
            ok += 1
        except Exception as e:
            print(f"ERR {p.name}: {type(e).__name__}: {e}")
            fail += 1

    print(f"\n완료: 성공 {ok} (캐시 {cached}) / 실패 {fail}")


if __name__ == "__main__":
    main()
