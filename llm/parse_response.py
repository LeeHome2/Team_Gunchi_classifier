"""
LLM JSON 응답 파싱 + 검증.

ai_server_1/llm/parse_response.py를 확장:
- 단일 bbox → bbox 배열(다중 평면도) 지원
- 하위 호환: 단일 bbox 응답이 와도 1개짜리 배열로 변환
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """응답 문자열에서 JSON 블록을 안전하게 뽑아낸다."""
    text = text.strip()

    # 순수 JSON인 경우
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 마크다운 코드블록/잡텍스트 섞인 경우: 첫 {..} 블록 추출
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("응답에서 JSON 객체를 찾지 못했습니다.")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 실패: {e}")


def _validate_single_bbox(bbox: Dict[str, Any]) -> Dict[str, float]:
    """단일 bbox 딕셔너리 검증. 픽셀 좌표 오면 자동 정규화 시도."""
    required = ["x_min", "y_min", "x_max", "y_max"]
    for key in required:
        if key not in bbox:
            raise ValueError(f"bbox에 '{key}' 키가 없습니다.")

    x_min = float(bbox["x_min"])
    y_min = float(bbox["y_min"])
    x_max = float(bbox["x_max"])
    y_max = float(bbox["y_max"])

    # 범위 초과 시 픽셀 좌표로 간주하고 최대값으로 정규화 시도
    max_val = max(x_min, y_min, x_max, y_max)
    if max_val > 1.0:
        # 일반 이미지 크기 범위(100~10000)면 픽셀 좌표로 간주
        # 가장 큰 값을 기준으로 정규화 (대충이라도 구조는 살림)
        scale = max_val if max_val > 1.0 else 1.0
        # 가로/세로 따로 추정: x는 x_max, y는 y_max 기준
        x_scale = x_max if x_max > 1.0 else 1.0
        y_scale = y_max if y_max > 1.0 else 1.0
        x_min = x_min / x_scale if x_scale > 1.0 else x_min
        x_max = x_max / x_scale if x_scale > 1.0 else x_max
        y_min = y_min / y_scale if y_scale > 1.0 else y_min
        y_max = y_max / y_scale if y_scale > 1.0 else y_max

    # 여전히 범위 밖이면 clip
    x_min = max(0.0, min(1.0, x_min))
    y_min = max(0.0, min(1.0, y_min))
    x_max = max(0.0, min(1.0, x_max))
    y_max = max(0.0, min(1.0, y_max))

    if x_min >= x_max or y_min >= y_max:
        raise ValueError(f"bbox 좌표 순서 오류 (정규화 후): {bbox}")

    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def validate_floorplan_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    평면도 bbox 검출 응답 검증.

    Supports two shapes:
    - 신형 (권장):
        {"floorplans_found": true, "floorplans": [{"label": "1F", "reason": "...", "bbox": {...}}, ...]}
    - 구형 (하위호환):
        {"floorplan_found": true, "reason": "...", "bbox": {...}}
      → 신형으로 정규화.
    """
    # 신형 응답
    if "floorplans" in data:
        found = bool(data.get("floorplans_found", len(data["floorplans"]) > 0))
        items = []
        for idx, fp in enumerate(data["floorplans"]):
            if "bbox" not in fp:
                raise ValueError(f"floorplans[{idx}]에 bbox 키가 없습니다.")
            # floor_index 파싱: 없으면 -1, 있으면 int 변환
            floor_index_raw = fp.get("floor_index")
            if floor_index_raw is not None:
                try:
                    floor_index = int(floor_index_raw)
                except (ValueError, TypeError):
                    floor_index = -1
            else:
                floor_index = -1
            items.append({
                "label": str(fp.get("label", f"floorplan_{idx}")),
                "floor_index": floor_index,
                "reason": str(fp.get("reason", "")),
                "bbox": _validate_single_bbox(fp["bbox"]),
            })
        return {"floorplans_found": found, "floorplans": items}

    # 구형 응답 (단일 bbox) → 1개짜리 배열로 변환
    if "bbox" in data:
        found = bool(data.get("floorplan_found", False))
        if not found:
            return {"floorplans_found": False, "floorplans": []}
        return {
            "floorplans_found": True,
            "floorplans": [{
                "label": "floorplan_0",
                "floor_index": -1,  # 구형 응답은 층 정보 없음
                "reason": str(data.get("reason", "")),
                "bbox": _validate_single_bbox(data["bbox"]),
            }],
        }

    raise ValueError("응답에 'floorplans' 또는 'bbox' 키가 없습니다.")
