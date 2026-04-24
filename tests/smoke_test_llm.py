"""
학과 vLLM 서버 연결 스모크 테스트.

세 모델 각각 가장 작은 요청으로 호출 → 성공/실패 리포트.
학과 서버(ceprj2.gachon.ac.kr)에서 실행 권장.

사용:
    python -m tests.smoke_test_llm
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.client import ask_text, ask_vision, get_embedding  # noqa: E402


def banner(name: str) -> None:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")


def test_text() -> bool:
    banner("[1/3] TEXT 모델 (qwen3.5-35b)")
    try:
        t0 = time.time()
        response = ask_text(
            prompt="'OK'라고만 짧게 답해주세요.",
            system="You are a helpful assistant. Reply with a single word.",
        )
        dt = time.time() - t0
        print(f"  응답 ({dt:.2f}s): {response!r}")
        print(f"  PASS")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


def test_embedding() -> bool:
    banner("[2/3] EMBEDDING 모델 (nomic-embed)")
    try:
        t0 = time.time()
        vec = get_embedding("wall")
        dt = time.time() - t0
        print(f"  응답 ({dt:.2f}s): 차원={len(vec)}, 앞 3개={vec[:3]}")
        assert len(vec) > 100, "임베딩 차원이 비정상"
        print(f"  PASS")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


def test_vision() -> bool:
    banner("[3/3] VISION 모델 (qwen3.5-35b multimodal)")
    # 작은 테스트 이미지 생성 (흰 배경에 'FLOOR PLAN' 텍스트)
    try:
        from PIL import Image, ImageDraw, ImageFont

        tmp = Path("/tmp/smoke_floorplan.png")
        img = Image.new("RGB", (400, 300), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # 단순한 집 모양 + 라벨
        draw.rectangle([60, 80, 340, 240], outline="black", width=3)
        draw.rectangle([80, 100, 180, 200], outline="black", width=2)
        draw.rectangle([200, 100, 320, 200], outline="black", width=2)
        draw.text((140, 260), "FLOOR PLAN", fill="black")
        img.save(tmp)

        t0 = time.time()
        response = ask_vision(
            image_path=tmp,
            prompt=(
                "이 이미지에 평면도(floor plan)가 있으면 true, 없으면 false로 답해라. "
                'JSON 형식: {"has_floorplan": true/false}'
            ),
            system="반드시 JSON으로만 응답한다.",
            response_format_json=True,
        )
        dt = time.time() - t0
        print(f"  응답 ({dt:.2f}s): {response!r}")
        print(f"  PASS")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


def main() -> int:
    print("학과 vLLM 프록시 스모크 테스트")
    print("BASE_URL: http://cellm.gachon.ac.kr:8000/v1")
    results = {
        "text": test_text(),
        "embedding": test_embedding(),
        "vision": test_vision(),
    }

    print(f"\n{'=' * 60}\n요약\n{'=' * 60}")
    for name, ok in results.items():
        icon = "PASS" if ok else "FAIL"
        print(f"  [{icon}] {name}")

    all_ok = all(results.values())
    print(f"\n전체: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
