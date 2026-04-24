"""
FastAPI 추론 서버.

CPU 전용 (학과 서버 GPU 독점 금지 규정 준수). 학습 프로그램과 분리.

엔드포인트:
  GET  /health
  POST /api/classify                 ← building_cesium이 호출
  GET  /api/mlops/experiments        ← 학습 이력
  GET  /api/mlops/experiments/{id}   ← 상세 (metrics 포함)
  GET  /api/mlops/models/active      ← 현재 active 모델
  POST /api/mlops/deploy             ← 특정 run_id를 active로 승격

실행:
  uvicorn main:app --host 0.0.0.0 --port 8001
  또는 python main.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SERVING_HOST, SERVING_PORT  # noqa: E402
from mlops.db import init_db  # noqa: E402
from mlops.registry import (  # noqa: E402
    get_active,
    get_experiment,
    list_experiments,
    log_predictions,
    set_active,
)


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("ai_layer_classifier")


# ─── FastAPI 설정 ────────────────────────────────────────────
app = FastAPI(
    title="AI Layer Classifier",
    description="DXF 엔티티 4-class 분류 + MLOps 레지스트리",
    version="0.1.0",
)

# CORS: AWS building_cesium (과 프론트 dev) 에서 호출 허용
cors_env = os.getenv("CORS_ORIGINS", "*")
cors_origins = ["*"] if cors_env.strip() == "*" else [o.strip() for o in cors_env.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_env.strip() != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    init_db()
    logger.info("MLOps DB initialized")
    # Active 모델 있으면 미리 로드 (첫 요청 지연 방지)
    try:
        from inference.predictor import get_active_bundle
        bundle = get_active_bundle()
        logger.info(f"Active model loaded: run_id={bundle.run_id}")
    except Exception as e:
        logger.warning(f"Active 모델 로드 실패 (학습 전이면 정상): {e}")


# ─── 요청/응답 스키마 ─────────────────────────────────────────
class Entity(BaseModel):
    entity_id: str | None = None
    entity_type: str | None = None
    raw_layer: str | None = None
    length: float | None = None
    bbox_width: float | None = None
    bbox_height: float | None = None
    aspect_ratio: float | None = None
    # 추가 필드는 무시됨


class ClassifyRequest(BaseModel):
    file_id: str | None = None
    entities: list[dict]          # building_cesium이 보내는 원본 형태 그대로
    log_predictions: bool = True  # DB에 샘플링 로그 남길지


class DeployRequest(BaseModel):
    run_id: str
    environment: str = "production"
    notes: str = ""


# ─── 엔드포인트 ──────────────────────────────────────────────
@app.get("/")
async def root():
    return {"service": "ai_layer_classifier", "version": app.version, "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/classify")
async def classify(req: ClassifyRequest):
    """엔티티 리스트 → 분류 결과."""
    from inference.predictor import classify_entities

    try:
        result = classify_entities(req.entities)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("classify 실패")
        raise HTTPException(status_code=500, detail=str(e))

    result["file_id"] = req.file_id

    # 샘플링 로그
    if req.log_predictions and result.get("predictions"):
        import random
        sample_rate = 0.2  # 20%
        sample = [p for p in result["predictions"] if random.random() < sample_rate]
        for p in sample:
            p["file_id"] = req.file_id
        try:
            log_predictions(result["model_version"], sample)
        except Exception as e:
            logger.warning(f"추론 로그 기록 실패: {e}")

    return result


# ─── MLOps 엔드포인트 ────────────────────────────────────────
@app.get("/api/mlops/experiments")
async def list_experiments_endpoint(limit: int = 50):
    return {"experiments": list_experiments(limit=limit)}


@app.get("/api/mlops/experiments/{run_id}")
async def get_experiment_endpoint(run_id: str):
    exp = get_experiment(run_id)
    if not exp:
        raise HTTPException(status_code=404, detail="실험을 찾을 수 없습니다")
    return exp


@app.get("/api/mlops/models/active")
async def get_active_model_endpoint():
    active = get_active()
    if not active:
        raise HTTPException(status_code=404, detail="활성 모델이 없습니다")
    return active


@app.post("/api/mlops/deploy")
async def deploy_endpoint(req: DeployRequest):
    # run_id 존재 여부 확인
    exp = get_experiment(req.run_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"run_id not found: {req.run_id}")

    set_active(req.run_id, environment=req.environment, notes=req.notes)

    # 새 active 모델 로드
    from inference.predictor import reload_active_bundle
    try:
        bundle = reload_active_bundle()
        return {
            "success": True,
            "active_run_id": bundle.run_id,
            "environment": req.environment,
        }
    except Exception as e:
        logger.exception("모델 재로드 실패")
        raise HTTPException(status_code=500, detail=f"deploy 실패: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=SERVING_HOST, port=SERVING_PORT, reload=False)
