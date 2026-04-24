"""
학과 TEXT LLM으로 레이어명을 4-class로 분류하는 능력 검증.

목적: 수동 라벨링 범위를 줄일 수 있는지 판단.
- 잘 되면 (>= 85% 정확도 추정) → weak label 생성에 LLM 활용, 수동은 Golden test만
- 안 되면 → 기존 키워드 매칭 + 수동 라벨링 계속

샘플: 데이터셋에서 실제 레이어명 25개 (다양한 표기 변이 포함)
평가: 호민님 눈으로 확인 + 합의된 정답과 비교

사용:
    python -m tests.test_layer_labeling_llm
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CLASSES  # noqa: E402
from llm.client import ask_text  # noqa: E402


# 실제 데이터셋에서 뽑은 샘플 레이어명 25개
# (데이터셋1-dxf 구성 분석 리포트 기반)
SAMPLE_LAYERS = [
    # wall 계열 (다양한 표기)
    "WALL",
    "wall",
    "A-WALL",
    "INTERNALWALL",
    "EXT WALLS",
    "Muro",
    "Compound Wall-ASAAS-0025",
    "Boundary Wall",
    "3BWALL",
    # door 계열
    "DOOR",
    "DOORS",
    "A-DOOR",
    "puerta",
    "sliding door",
    "D",
    # window 계열
    "WINDOW",
    "Windows",
    "WIN",
    "WINOWS",  # 오타
    "ventana",
    # other 계열 (wall 아닌 것 — 잘 구분해야 함)
    "DIMENSION",
    "FURNITURE",
    "TEXT",
    "Interior",  # mixed일 수도 있지만 레이어명만 보면 other에 가까움
    "Arch-plan",  # 전체 도면 — 애매함, other 처리 예상
]

# 호민님이 보고 합의할 '정답' — 이게 LLM 평가 기준
EXPECTED = {
    "WALL": "wall", "wall": "wall", "A-WALL": "wall", "INTERNALWALL": "wall",
    "EXT WALLS": "wall", "Muro": "wall", "Compound Wall-ASAAS-0025": "wall",
    "Boundary Wall": "wall", "3BWALL": "wall",
    "DOOR": "door", "DOORS": "door", "A-DOOR": "door", "puerta": "door",
    "sliding door": "door", "D": "door",
    "WINDOW": "window", "Windows": "window", "WIN": "window",
    "WINOWS": "window", "ventana": "window",
    "DIMENSION": "other", "FURNITURE": "other", "TEXT": "other",
    "Interior": "other", "Arch-plan": "other",
}


PROMPT_TEMPLATE = """당신은 CAD 도면의 레이어명을 보고 건축 요소 카테고리로 분류해야 합니다.

분류 옵션 (반드시 이 4개 중 하나):
- wall: 벽체 (외벽/내벽 구분 없음). 예: WALL, MURO, A-WALL, Boundary Wall
- door: 문. 예: DOOR, puerta, sliding door
- window: 창문. 예: WINDOW, ventana, WIN
- other: 위 셋이 아닌 모든 것 (치수, 텍스트, 가구, 계단, 기둥, 전체 도면 등)

주의사항:
- 레이어명은 영어/스페인어/한국어 등 다양한 언어일 수 있음
- 오타도 가능 (WINOWS = WINDOWS)
- 약어도 많음 (D = DOOR, WIN = WINDOW)
- 대소문자 무관
- "Interior", "Arch-plan" 같은 전체 도면 레이어는 other로 분류

레이어명 목록:
{layers}

JSON 형식으로만 응답하라. 다른 어떤 텍스트도 쓰지 말 것:
{{
  "classifications": [
    {{"layer": "레이어명 그대로", "class": "wall/door/window/other", "confidence": 0.0~1.0, "reason": "짧은 한 문장"}}
  ]
}}"""


def classify_layers(layer_names: List[str]) -> List[Dict]:
    """학과 TEXT LLM 호출해서 4-class 분류 결과 받기."""
    layers_text = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(layer_names))
    prompt = PROMPT_TEMPLATE.format(layers=layers_text)

    response_str = ask_text(
        prompt=prompt,
        system="너는 CAD 도면 레이어 분류 도우미다. 반드시 JSON으로만 응답한다.",
        response_format_json=True,
    )
    data = json.loads(response_str)
    return data.get("classifications", [])


def evaluate(predictions: List[Dict], expected: Dict[str, str]) -> Dict:
    """예측 vs 정답 비교 리포트."""
    correct = wrong = missing = 0
    wrong_items = []
    by_class_correct = Counter()
    by_class_total = Counter()

    for pred in predictions:
        layer = pred.get("layer", "")
        pred_class = pred.get("class", "")
        exp_class = expected.get(layer)
        if exp_class is None:
            # 정답 없음 — 평가 제외
            continue
        by_class_total[exp_class] += 1
        if pred_class == exp_class:
            correct += 1
            by_class_correct[exp_class] += 1
        else:
            wrong += 1
            wrong_items.append({
                "layer": layer,
                "expected": exp_class,
                "predicted": pred_class,
                "reason": pred.get("reason", ""),
                "confidence": pred.get("confidence"),
            })

    # 빠진 항목 확인
    returned_layers = {p.get("layer", "") for p in predictions}
    for layer in expected:
        if layer not in returned_layers:
            missing += 1

    total_evaluated = correct + wrong
    accuracy = correct / total_evaluated if total_evaluated > 0 else 0.0

    return {
        "total": len(expected),
        "evaluated": total_evaluated,
        "correct": correct,
        "wrong": wrong,
        "missing": missing,
        "accuracy": round(accuracy, 3),
        "per_class_accuracy": {
            c: round(by_class_correct[c] / by_class_total[c], 3) if by_class_total[c] > 0 else None
            for c in CLASSES
        },
        "per_class_total": dict(by_class_total),
        "wrong_items": wrong_items,
    }


def main() -> int:
    print("=" * 60)
    print("LLM 기반 레이어명 4-class 분류 실험")
    print("=" * 60)
    print(f"샘플 수: {len(SAMPLE_LAYERS)}")
    print(f"분류: {CLASSES}")

    print("\n학과 TEXT LLM 호출 중...")
    t0 = time.time()
    try:
        predictions = classify_layers(SAMPLE_LAYERS)
    except Exception as e:
        print(f"LLM 호출 실패: {type(e).__name__}: {e}")
        return 1
    dt = time.time() - t0
    print(f"응답 시간: {dt:.2f}s")
    print(f"받은 항목 수: {len(predictions)}")

    # 예측 결과 출력
    print("\n" + "-" * 60)
    print(f"{'레이어명':<35s} | {'예상':<7s} | {'예측':<7s} | 정오")
    print("-" * 60)
    for pred in predictions:
        layer = pred.get("layer", "?")
        pred_class = pred.get("class", "?")
        exp = EXPECTED.get(layer, "-")
        mark = "  OK" if pred_class == exp else "FAIL"
        print(f"{layer[:34]:<35s} | {exp:<7s} | {pred_class:<7s} | {mark}")

    # 평가
    report = evaluate(predictions, EXPECTED)
    print("\n" + "=" * 60)
    print("평가 요약")
    print("=" * 60)
    print(f"  전체 정확도: {report['accuracy']:.1%} ({report['correct']}/{report['evaluated']})")
    print(f"  클래스별 정확도:")
    for cls, acc in report["per_class_accuracy"].items():
        total = report["per_class_total"].get(cls, 0)
        if total > 0:
            print(f"    {cls:<7s}: {acc:.1%} (n={total})")
    if report["missing"] > 0:
        print(f"  빠진 응답: {report['missing']}개")

    if report["wrong_items"]:
        print(f"\n틀린 항목 ({len(report['wrong_items'])}개):")
        for item in report["wrong_items"]:
            conf = f", conf={item['confidence']:.2f}" if item.get("confidence") else ""
            print(f"  - [{item['layer']}] 예상={item['expected']}, 예측={item['predicted']}{conf}")
            print(f"    이유: {item['reason']}")

    # 권장 조치
    acc = report["accuracy"]
    print("\n" + "=" * 60)
    print("권장 조치")
    print("=" * 60)
    if acc >= 0.90:
        print("  [OK] 매우 좋음 → LLM으로 weak label 생성 확정. 수동은 Golden test만.")
    elif acc >= 0.80:
        print("  [~~] 쓸만함 → LLM weak label + 낮은 confidence만 수동 검수")
    elif acc >= 0.65:
        print("  [??] 애매 → LLM + 기존 키워드 매칭 앙상블, 충분한 수동 검수 필요")
    else:
        print("  [NG] 부족 → 기존 키워드 매칭 + 수동 라벨링 계속, LLM 사용 재검토")

    # JSON 결과 저장
    out = Path(__file__).resolve().parent.parent / "data" / "reports" / "llm_labeling_experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "predictions": predictions,
                "report": report,
                "elapsed_seconds": dt,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n결과 저장: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
