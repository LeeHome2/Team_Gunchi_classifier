"""
학과 vLLM 프록시 클라이언트.

OpenAI 호환 API를 사용하며 세 가지 모델을 감싼다:
- text: 텍스트 생성/분석
- vision: 이미지 + 텍스트
- embedding: 텍스트 벡터화 (nomic-embed-text-v1.5)

ai_server_1/llm/client.py 가 OpenAI GPT-4o-mini를 쓰던 것을
학과 vLLM 서버로 교체한 버전.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Optional, Union

from openai import OpenAI

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_EMBEDDING_MODEL,
    LLM_TEXT_MODEL,
    LLM_TIMEOUT,
    LLM_VISION_MODEL,
)


def _client() -> OpenAI:
    """OpenAI 호환 클라이언트. 학과 프록시로 향함."""
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT)


# ─── 이미지 헬퍼 ─────────────────────────────────────────────
def encode_image_to_base64(image_path: Union[str, Path]) -> str:
    """PNG/JPG 파일을 base64 문자열로 인코딩."""
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"이미지 파일이 없습니다: {p}")
    return base64.b64encode(p.read_bytes()).decode("utf-8")


# ─── Vision 모델 ─────────────────────────────────────────────
def ask_vision(
    image_path: Union[str, Path],
    prompt: str,
    system: Optional[str] = None,
    *,
    model: Optional[str] = None,
    response_format_json: bool = True,
) -> str:
    """
    이미지 + 프롬프트 → 모델 응답 문자열.

    기본적으로 JSON 응답을 강제한다(response_format_json=True).
    """
    image_b64 = encode_image_to_base64(image_path)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ],
    })

    kwargs = {
        "model": model or LLM_VISION_MODEL,
        "messages": messages,
    }
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}

    response = _client().chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ─── Text 모델 ───────────────────────────────────────────────
def ask_text(
    prompt: str,
    system: Optional[str] = None,
    *,
    model: Optional[str] = None,
    response_format_json: bool = False,
) -> str:
    """단순 텍스트 요청."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model or LLM_TEXT_MODEL,
        "messages": messages,
    }
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}

    response = _client().chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ─── Embedding 모델 ──────────────────────────────────────────
def get_embedding(text: str, *, model: Optional[str] = None) -> List[float]:
    """단일 문자열 → 벡터(리스트). 레이어명 feature 생성에 사용."""
    response = _client().embeddings.create(
        model=model or LLM_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def get_embeddings_batch(
    texts: List[str],
    *,
    model: Optional[str] = None,
) -> List[List[float]]:
    """여러 문자열을 한 번에 임베딩. 토큰 상한 주의 (32k)."""
    if not texts:
        return []
    response = _client().embeddings.create(
        model=model or LLM_EMBEDDING_MODEL,
        input=texts,
    )
    # 응답 순서 보장
    return [item.embedding for item in response.data]
