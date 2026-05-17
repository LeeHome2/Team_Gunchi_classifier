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
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SERVING_HOST, SERVING_PORT  # noqa: E402
from mlops.db import init_db  # noqa: E402
from mlops.registry import (  # noqa: E402
    get_active,
    get_experiment,
    list_experiments,
    log_predictions,
    register_experiment,
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
# Python 3.9 호환을 위해 Optional/List 사용 (str | None 같은 PEP 604 문법 회피)
class Entity(BaseModel):
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    raw_layer: Optional[str] = None
    length: Optional[float] = None
    bbox_width: Optional[float] = None
    bbox_height: Optional[float] = None
    aspect_ratio: Optional[float] = None
    # 추가 필드는 무시됨


class ClassifyRequest(BaseModel):
    file_id: Optional[str] = None
    entities: List[dict]          # building_cesium이 보내는 원본 형태 그대로
    log_predictions: bool = True  # DB에 샘플링 로그 남길지


class DeployRequest(BaseModel):
    run_id: str
    environment: str = "production"
    notes: Optional[str] = None


class TrainRequest(BaseModel):
    run_id: Optional[str] = None
    max_iter: int = 200
    max_depth: int = 7
    learning_rate: float = 0.08
    input_dir: Optional[str] = None  # 기본: data/labeled
    # 분할 비율 (사용자 설정 가능). test_ratio = 1 - train_ratio - val_ratio
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # 모델 타입 선택: hist_gradient, random_forest, xgboost
    model_type: str = "hist_gradient"


class CollectRequest(BaseModel):
    dxf_dir: Optional[str] = None    # 기본: ~/데이터셋1-dxf/dxf
    mock: bool = False               # vLLM 호출 없이 mock
    limit: Optional[int] = None      # 처리 개수 제한 (디버그용)


# ─── MLOps 대시보드 HTML ─────────────────────────────────────
DASHBOARD_HTML = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>AI Layer Classifier · MLOps Console</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Apple SD Gothic Neo", sans-serif;
         background: #0a0e27; color: #e0e0e0; margin: 0; padding: 24px; }
  h1 { color: #4ade80; margin-top: 0; }
  h2 { color: #f5f5f5; border-bottom: 1px solid #333; padding-bottom: 8px; }
  .card { background: #1a1f3a; padding: 20px; border-radius: 8px; margin: 16px 0;
          box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #2a2f4a; }
  th { color: #888; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
  td { font-size: 13px; }
  tr:hover { background: #232847; }
  .badge { padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: 700; }
  .badge-active { background: #4ade80; color: #0a0e27; }
  .badge-completed { background: #3b82f6; color: white; }
  .badge-failed { background: #ef4444; color: white; }
  button { background: #4ade80; color: #0a0e27; border: none; padding: 6px 14px;
           border-radius: 4px; cursor: pointer; font-weight: 600; font-size: 12px; }
  button:hover { background: #22c55e; }
  button:disabled { background: #444; color: #888; cursor: not-allowed; }
  .metric { color: #fbbf24; font-weight: 600; }
  pre { background: #000; padding: 14px; border-radius: 4px; overflow-x: auto;
        font-size: 12px; max-height: 400px; }
  a { color: #60a5fa; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .links { display: flex; gap: 16px; margin: 8px 0 24px; flex-wrap: wrap; }
  .links a { padding: 6px 12px; background: #1a1f3a; border-radius: 4px; }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
  .stat { background: #232847; padding: 12px 16px; border-radius: 6px; }
  .stat-label { font-size: 11px; color: #888; text-transform: uppercase; }
  .stat-value { font-size: 20px; font-weight: 700; color: #4ade80; margin-top: 4px; }
  .footer { color: #666; font-size: 11px; text-align: center; margin-top: 32px; }
</style>
</head>
<body>
<h1>🤖 AI Layer Classifier · MLOps Console</h1>
<div class="links">
  <a href="/docs" target="_blank">📘 Swagger API Docs</a>
  <a href="/api/mlops/experiments" target="_blank">📊 Experiments JSON</a>
  <a href="/api/mlops/models/active" target="_blank">⚡ Active Model JSON</a>
  <a href="/health" target="_blank">💚 Health</a>
</div>

<div class="card">
  <h2>⚡ Active Model</h2>
  <div id="active">로딩 중...</div>
</div>

<div class="card">
  <h2>📦 Datasets &amp; Splits</h2>
  <p style="color:#888;font-size:12px;margin-top:0">
    데이터셋 등록 / 버전 관리 / 학습·검증·테스트 분할 (진도표 v0.2 항목)
  </p>
  <div id="datasets">로딩 중...</div>
</div>

<div class="card">
  <h2>📋 Experiments</h2>
  <table id="exp-table">
    <thead>
      <tr><th>Run ID</th><th>Model Type</th><th>Status</th><th>Created</th><th>Test F1</th><th>Action</th></tr>
    </thead>
    <tbody><tr><td colspan="6">로딩 중...</td></tr></tbody>
  </table>
</div>

<div class="card">
  <h2>🧪 Quick Classification Test</h2>
  <p style="color:#888;font-size:13px;margin-top:0">
    학습 시 본 표기 (WALL/DOOR/WINDOW) + 못 본 표기 (BoundaryWall_Main) 혼합 샘플
  </p>
  <button onclick="testClassify()">▶ Run Sample Classification</button>
  <pre id="classify-result">결과가 여기 표시됩니다</pre>
</div>

<div class="footer">
  Auto-refresh every 5s · Team 건치 · 2026 종합설계프로젝트
</div>

<script>
async function loadActive() {
  try {
    const r = await fetch('/api/mlops/models/active');
    if (!r.ok) throw new Error('no active');
    const d = await r.json();
    document.getElementById('active').innerHTML = `
      <div class="stat-grid">
        <div class="stat"><div class="stat-label">Run ID</div><div class="stat-value" style="font-size:13px;font-family:monospace">${d.run_id}</div></div>
        <div class="stat"><div class="stat-label">Model Type</div><div class="stat-value" style="font-size:14px">${d.model_type}</div></div>
        <div class="stat"><div class="stat-label">Environment</div><div class="stat-value" style="font-size:14px">${d.environment}</div></div>
        <div class="stat"><div class="stat-label">Deployed At</div><div class="stat-value" style="font-size:13px">${d.deployed_at}</div></div>
      </div>`;
  } catch (e) {
    document.getElementById('active').innerHTML = '<span style="color:#ef4444">❌ 활성 모델 없음 — 학습 후 deploy 필요</span>';
  }
}

async function loadDatasets() {
  try {
    const r = await fetch('/api/mlops/datasets');
    const d = await r.json();
    const stagesHtml = d.stages.map(s => {
      const lastMod = s.last_modified
        ? new Date(s.last_modified * 1000).toISOString().slice(0, 19).replace('T', ' ')
        : '—';
      const okBadge = s.exists && s.count > 0
        ? '<span class="badge badge-active">OK</span>'
        : '<span class="badge" style="background:#444;color:#aaa">EMPTY</span>';
      return `
        <tr>
          <td>${s.label}</td>
          <td style="text-align:right"><span class="metric">${s.count}</span> 개</td>
          <td style="text-align:right">${s.size_mb} MB</td>
          <td>${okBadge}</td>
          <td style="color:#888;font-size:11px">${lastMod}</td>
        </tr>`;
    }).join('');

    let splitHtml = '';
    if (d.latest_split) {
      const ls = d.latest_split;
      splitHtml = `
        <div style="margin-top:12px">
          <div style="font-size:13px;color:#888;margin-bottom:6px">
            📊 가장 최근 학습의 train/val/test 분할 — <span style="font-family:monospace;color:#60a5fa">${ls.run_id}</span>
          </div>
          <div class="stat-grid" style="grid-template-columns:repeat(3,1fr)">
            <div class="stat">
              <div class="stat-label">Train</div>
              <div class="stat-value" style="font-size:18px">${ls.train_files ?? '?'} 파일</div>
              <div style="font-size:11px;color:#888">${(ls.train_rows ?? 0).toLocaleString()} rows</div>
            </div>
            <div class="stat">
              <div class="stat-label">Val</div>
              <div class="stat-value" style="font-size:18px">${ls.val_files ?? '?'} 파일</div>
              <div style="font-size:11px;color:#888">${(ls.val_rows ?? 0).toLocaleString()} rows</div>
            </div>
            <div class="stat">
              <div class="stat-label">Test</div>
              <div class="stat-value" style="font-size:18px">${ls.test_files ?? '?'} 파일</div>
              <div style="font-size:11px;color:#888">${(ls.test_rows ?? 0).toLocaleString()} rows</div>
            </div>
          </div>
        </div>`;
    } else {
      splitHtml = '<div style="margin-top:12px;color:#888;font-size:13px">아직 학습 이력 없음</div>';
    }

    let metaHtml = '';
    const datasets = d.meta?.datasets || [];
    if (datasets.length > 0) {
      metaHtml = `
        <div style="margin-top:12px;font-size:12px;color:#888">
          📚 등록된 데이터셋 (${datasets.length}개): ${datasets.map(ds => `<code style="background:#000;padding:2px 6px;border-radius:3px">${ds.id || ds.name || '-'}</code>`).join(' · ')}
        </div>`;
    }

    document.getElementById('datasets').innerHTML = `
      <table>
        <thead>
          <tr><th>단계</th><th style="text-align:right">개수</th><th style="text-align:right">크기</th><th>상태</th><th>마지막 수정</th></tr>
        </thead>
        <tbody>${stagesHtml}</tbody>
      </table>
      ${splitHtml}
      ${metaHtml}
    `;
  } catch (e) {
    document.getElementById('datasets').innerHTML = '<span style="color:#ef4444">데이터셋 정보 로드 실패</span>';
  }
}

async function loadExperiments() {
  const r = await fetch('/api/mlops/experiments');
  const data = await r.json();
  const tbody = document.querySelector('#exp-table tbody');
  let activeId = null;
  try {
    const a = await fetch('/api/mlops/models/active');
    if (a.ok) activeId = (await a.json()).run_id;
  } catch(e) {}

  if (data.experiments.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="color:#888">실험 없음 — python -m training.train 으로 학습 시작</td></tr>';
    return;
  }

  tbody.innerHTML = '';
  for (const exp of data.experiments) {
    const isActive = exp.run_id === activeId;
    let f1 = '-';
    try {
      const m = await fetch(`/api/mlops/experiments/${exp.run_id}`);
      if (m.ok) {
        const md = await m.json();
        const test = md.metrics?.test;
        if (test) f1 = `<span class="metric">${test.f1_macro?.toFixed(4) ?? '-'}</span>`;
      }
    } catch(e) {}
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="font-family:monospace;font-size:11px"><a href="/api/mlops/experiments/${exp.run_id}" target="_blank">${exp.run_id}</a></td>
      <td>${exp.model_type}</td>
      <td><span class="badge badge-${isActive ? 'active' : (exp.status==='failed'?'failed':'completed')}">${isActive ? 'ACTIVE' : exp.status.toUpperCase()}</span></td>
      <td style="color:#888">${exp.created_at}</td>
      <td>${f1}</td>
      <td><button ${isActive ? 'disabled' : ''} onclick="deployModel('${exp.run_id}')">${isActive ? 'Active' : 'Deploy'}</button></td>`;
    tbody.appendChild(tr);
  }
}

async function deployModel(runId) {
  if (!confirm(`모델을 ${runId}로 교체할까요?`)) return;
  const r = await fetch('/api/mlops/deploy', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({run_id: runId, environment: 'production', notes: 'dashboard manual deploy'}),
  });
  if (r.ok) {
    alert('✅ 배포 완료!');
    refreshAll();
  } else {
    const e = await r.json();
    alert('❌ 실패: ' + JSON.stringify(e));
  }
}

async function testClassify() {
  document.getElementById('classify-result').textContent = '추론 중...';
  const sample = {
    file_id: 'dashboard_test',
    entities: [
      {entity_id:'1', entity_type:'LINE', raw_layer:'WALL', length:3.5, bbox_width:3.5, bbox_height:0.01, aspect_ratio:350},
      {entity_id:'2', entity_type:'ARC', raw_layer:'DOOR', length:1.5, bbox_width:0.9, bbox_height:0.9, aspect_ratio:1.0},
      {entity_id:'3', entity_type:'LINE', raw_layer:'WINDOW-ASAAS-0025', length:1.2, bbox_width:1.2, bbox_height:0.01, aspect_ratio:120},
      {entity_id:'4', entity_type:'LINE', raw_layer:'BoundaryWall_Main', length:5.0, bbox_width:5.0, bbox_height:0.01, aspect_ratio:500},
      {entity_id:'5', entity_type:'TEXT', raw_layer:'DIMENSIONS', length:0, bbox_width:2, bbox_height:0.3, aspect_ratio:6.7},
    ],
    log_predictions: false
  };
  const r = await fetch('/api/classify', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(sample),
  });
  const data = await r.json();
  document.getElementById('classify-result').textContent = JSON.stringify(data, null, 2);
}

function refreshAll() { loadActive(); loadDatasets(); loadExperiments(); }
refreshAll();
setInterval(refreshAll, 5000);
</script>
</body>
</html>
"""


# ─── 엔드포인트 ──────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """MLOps 대시보드 (브라우저용)."""
    return DASHBOARD_HTML


@app.get("/info")
async def info():
    """서비스 정보 (JSON)."""
    return {"service": "ai_layer_classifier", "version": app.version, "docs": "/docs", "dashboard": "/"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/detect-floorplan")
async def detect_floorplan_endpoint(
    file: UploadFile = File(...),
    mock: bool = Form(False),
):
    """
    DXF 파일 → vLLM Vision 평면도 bbox 검출.

    응답:
      {
        "floorplans_found": true,
        "floorplans": [
          {
            "label": "1F",        # vLLM 추론 (1F/2F/B1/RF 등)
            "floor_index": 0,     # 0-based (LLM 추론, 실패 시 -1)
            "reason": "...",
            "bbox": {"x_min": 0.05, "y_min": 0.10, "x_max": 0.45, "y_max": 0.50}
          },
          ...
        ],
        "extent_dxf": {"min_x": ..., "min_y": ..., "max_x": ..., "max_y": ...}
        # DXF 의 실제 좌표 범위 — 정규화 bbox → DXF 좌표 환산용
      }
    """
    import shutil
    import tempfile

    import ezdxf

    from config import BASE_DIR
    from dataset.detect_floorplan import detect_floorplan_for_file
    from dataset.render_preview import render_dxf_to_png

    # 1. 업로드 임시 저장
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        dxf_path = Path(tmp.name)

    try:
        # 2. PNG 렌더 (캐시 디렉토리)
        render_dir = BASE_DIR / "data" / "preview"
        render_dir.mkdir(parents=True, exist_ok=True)
        meta = render_dxf_to_png(dxf_path, render_dir)
        png_path = Path(meta["png_path"])

        # 3. vLLM 검출
        result = detect_floorplan_for_file(
            png_path=png_path,
            cache_dir=BASE_DIR / "data" / "processed",
            mock=mock,
            use_cache=True,
        )

        # 4. DXF extent (정규화 → DXF 환산용)
        if meta.get("extents_cad"):
            result["extent_dxf"] = meta["extents_cad"]
        else:
            result["extent_dxf"] = _compute_dxf_extent(dxf_path)

        return result
    finally:
        try:
            dxf_path.unlink(missing_ok=True)
        except Exception:
            pass


def _compute_dxf_extent(dxf_path: Path) -> dict:
    """DXF 파일에서 좌표 범위 계산."""
    import ezdxf

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    xs, ys = [], []
    for ent in msp:
        et = ent.dxftype()
        if et == "LINE":
            xs.extend([ent.dxf.start[0], ent.dxf.end[0]])
            ys.extend([ent.dxf.start[1], ent.dxf.end[1]])
        elif et == "LWPOLYLINE":
            for p in ent.get_points():
                xs.append(p[0])
                ys.append(p[1])
    if not xs:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}
    return {
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
    }


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
@app.get("/api/mlops/datasets")
async def list_datasets_endpoint(dataset_id: Optional[str] = None):
    """
    학습 데이터셋 단계별 통계 + 가장 최근 학습의 train/val/test split 정보.

    Query 파라미터:
      - dataset_id: 특정 데이터셋 ID. 지정하면 해당 데이터셋의 stages만 반환.

    진도표 v0.2 항목 시각화:
      - "학습 데이터셋 등록 기능"      : raw_dxf, processed, labeled 단계별 통계
      - "데이터셋 버전관리 기능"        : configs/dataset_meta.json
      - "학습/검증/테스트 분할 생성 기능" : latest_split (가장 최근 experiment)
    """
    import json
    from pathlib import Path
    from config import (
        BASE_DIR,
        DATASET_META_PATH,
        EXTERNAL_DATASET_DIR,
    )

    DATA_DIR = BASE_DIR / "data"

    def stage_stats(folder: Path, pattern: str, label: str):
        """단계별 폴더의 파일 수, 사이즈, 마지막 수정 시각."""
        if not folder.exists():
            return {
                "label": label,
                "path": str(folder),
                "exists": False,
                "count": 0,
                "size_mb": 0.0,
                "last_modified": None,
            }
        files = list(folder.glob(pattern))
        size_bytes = sum(f.stat().st_size for f in files if f.is_file())
        return {
            "label": label,
            "path": str(folder),
            "exists": True,
            "count": len(files),
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "last_modified": (
                max((f.stat().st_mtime for f in files if f.is_file()), default=None)
            ),
        }

    def stage_stats_for_dataset(dxf_dir: Path, dataset_name: str):
        """
        특정 데이터셋 디렉토리 기준 파이프라인 단계 계산.

        두 가지 디렉토리 구조 지원:
        1. 하위 디렉토리 구조: data/processed/{dataset_name}/*.csv
        2. 플랫 구조: data/processed/*.csv (DXF 파일명으로 매칭)
        """
        # DXF 파일명 목록 (확장자 제외)
        dxf_basenames = set()
        if dxf_dir.exists():
            dxf_basenames = {f.stem for f in dxf_dir.glob("*.dxf")}

        def stage_stats_filtered(folder: Path, pattern: str, label: str):
            """플랫 구조에서 DXF 파일명 기준으로 필터링된 통계."""
            if not folder.exists():
                return {
                    "label": label,
                    "path": str(folder),
                    "exists": False,
                    "count": 0,
                    "size_mb": 0.0,
                    "last_modified": None,
                }

            # 패턴으로 모든 파일 찾기
            all_files = list(folder.glob(pattern))

            # DXF 파일명과 매칭되는 파일만 필터링
            # 예: "file1.csv" -> "file1"과 DXF 파일명 비교
            # 크롭/라벨 파일: "file1__floorplan_0.csv" -> "file1"
            import re
            matched_files = []
            for f in all_files:
                # 파일명에서 확장자들 제거 (.bboxes.json, .meta.json, .csv 등)
                base = f.stem
                for suffix in [".bboxes", ".meta", ".crop"]:
                    if base.endswith(suffix):
                        base = base[:-len(suffix)]
                        break

                # 크롭/라벨 파일의 __floorplan_N 또는 __full 접미사 제거
                base = re.sub(r'__(?:floorplan_\d+|full)$', '', base)

                if base in dxf_basenames:
                    matched_files.append(f)

            if not matched_files:
                return {
                    "label": label,
                    "path": str(folder),
                    "exists": True,
                    "count": 0,
                    "size_mb": 0.0,
                    "last_modified": None,
                }

            size_bytes = sum(f.stat().st_size for f in matched_files if f.is_file())
            return {
                "label": label,
                "path": str(folder),
                "exists": True,
                "count": len(matched_files),
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "last_modified": (
                    max((f.stat().st_mtime for f in matched_files if f.is_file()), default=None)
                ),
            }

        # 하위 디렉토리 구조 확인
        processed_subdir = DATA_DIR / "processed" / dataset_name
        labeled_subdir = DATA_DIR / "labeled" / dataset_name
        preview_subdir = DATA_DIR / "preview" / dataset_name
        cropped_subdir = DATA_DIR / "cropped" / dataset_name

        # 하위 디렉토리가 존재하면 기존 방식, 아니면 플랫 구조에서 필터링
        use_subdir = processed_subdir.exists() and any(processed_subdir.glob("*.csv"))

        if use_subdir:
            return [
                stage_stats(dxf_dir, "*.dxf", "1) 원본 DXF"),
                stage_stats(processed_subdir, "*.csv", "2) parse_dxf 결과 CSV"),
                stage_stats(processed_subdir, "*.bboxes.json", "3) vLLM Vision bbox JSON"),
                stage_stats(preview_subdir, "*.png", "4) 렌더 PNG"),
                stage_stats(cropped_subdir, "*.csv", "5) bbox 크롭 CSV"),
                stage_stats(labeled_subdir, "*.csv", "6) 학습용 라벨링 CSV ⭐"),
            ]
        else:
            # 플랫 구조: DXF 파일명으로 매칭
            return [
                stage_stats(dxf_dir, "*.dxf", "1) 원본 DXF"),
                stage_stats_filtered(DATA_DIR / "processed", "*.csv", "2) parse_dxf 결과 CSV"),
                stage_stats_filtered(DATA_DIR / "processed", "*.bboxes.json", "3) vLLM Vision bbox JSON"),
                stage_stats_filtered(DATA_DIR / "preview", "*.png", "4) 렌더 PNG"),
                stage_stats_filtered(DATA_DIR / "cropped", "*.csv", "5) bbox 크롭 CSV"),
                stage_stats_filtered(DATA_DIR / "labeled", "*.csv", "6) 학습용 라벨링 CSV ⭐"),
            ]

    # 데이터셋 메타 (configs/dataset_meta.json)
    meta = {"datasets": []}
    if DATASET_META_PATH.exists():
        try:
            meta = json.loads(DATASET_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    # dataset_id가 지정된 경우 해당 데이터셋만
    if dataset_id:
        target = next(
            (ds for ds in meta.get("datasets", []) if ds.get("id") == dataset_id),
            None,
        )
        # dxf_dir 또는 path 키 모두 지원 (호환성)
        dxf_path = target.get("dxf_dir") or target.get("path") if target else None
        if target and dxf_path:
            dxf_dir = Path(dxf_path)
            # 상대 경로인 경우 BASE_DIR 기준으로 변환
            if not dxf_dir.is_absolute():
                dxf_dir = (BASE_DIR / dxf_dir).resolve()
            dataset_name = target.get("name") or target.get("id") or dxf_dir.name
            stages = stage_stats_for_dataset(dxf_dir, dataset_name)
        else:
            # 해당 데이터셋을 찾을 수 없는 경우 빈 stages
            stages = []
    else:
        # 전체 stages (기존 동작)
        stages = [
            stage_stats(EXTERNAL_DATASET_DIR, "*.dxf", "1) 원본 DXF (외부 데이터셋)"),
            stage_stats(DATA_DIR / "processed", "*.csv", "2) parse_dxf 결과 CSV"),
            stage_stats(DATA_DIR / "processed", "*.bboxes.json", "3) vLLM Vision bbox JSON"),
            stage_stats(DATA_DIR / "preview", "*.png", "4) 렌더 PNG"),
            stage_stats(DATA_DIR / "cropped", "*.csv", "5) bbox 크롭 CSV"),
            stage_stats(DATA_DIR / "labeled", "*.csv", "6) 학습용 라벨링 CSV ⭐"),
        ]

    # 가장 최근 experiment의 train/val/test split 정보
    latest_split = None
    try:
        exps = list_experiments(limit=1)
        if exps:
            exp = get_experiment(exps[0]["run_id"])
            if exp:
                ti = exp.get("train_info") or {}
                if isinstance(ti, str):
                    try:
                        ti = json.loads(ti)
                    except Exception:
                        ti = {}
                latest_split = {
                    "run_id": exps[0]["run_id"],
                    "created_at": exps[0].get("created_at"),
                    "train_files": (
                        len(ti.get("train_files", []))
                        if isinstance(ti.get("train_files"), list)
                        else None
                    ),
                    "val_files": (
                        len(ti.get("val_files", []))
                        if isinstance(ti.get("val_files"), list)
                        else None
                    ),
                    "test_files": (
                        len(ti.get("test_files", []))
                        if isinstance(ti.get("test_files"), list)
                        else None
                    ),
                    "train_rows": ti.get("train_rows"),
                    "val_rows": ti.get("val_rows"),
                    "test_rows": ti.get("test_rows"),
                    "training_time_seconds": ti.get("training_time_seconds"),
                }
    except Exception as e:
        logger.warning(f"latest split 조회 실패: {e}")

    # processed_datasets: 빌드별 라벨링 데이터셋 분리 목록
    processed_datasets = []
    labeled_dir = DATA_DIR / "labeled"
    if labeled_dir.exists():
        import re
        from datetime import datetime

        # 빌드별 디렉토리 또는 빌드 ID가 포함된 CSV 파일 탐색
        # 패턴: build_YYYYMMDD_HHMMSS_XXXXXX
        build_pattern = re.compile(r"^build_(\d{8}_\d{6})_([a-f0-9]{6})$")

        # 1. 빌드별 서브디렉토리 탐색
        for subdir in labeled_dir.iterdir():
            if subdir.is_dir():
                match = build_pattern.match(subdir.name)
                if match:
                    timestamp_str, short_id = match.groups()
                    csv_files = list(subdir.glob("*.csv"))
                    total_size = sum(f.stat().st_size for f in csv_files)
                    labeled_count = 0
                    for csv_file in csv_files:
                        try:
                            with open(csv_file, "r", encoding="utf-8") as f:
                                labeled_count += sum(1 for _ in f) - 1  # 헤더 제외
                        except Exception:
                            pass

                    try:
                        processed_at = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").isoformat()
                    except ValueError:
                        processed_at = None

                    processed_datasets.append({
                        "id": subdir.name,
                        "name": subdir.name,
                        "processed_at": processed_at,
                        "labeled_count": labeled_count,
                        "csv_path": str(subdir),
                        "size_mb": round(total_size / (1024 * 1024), 2),
                        "status": "ready" if csv_files else "empty",
                    })

        # 2. 플랫 구조에서 빌드 ID 없는 CSV도 하나의 "default" 데이터셋으로 집계
        flat_csvs = [f for f in labeled_dir.glob("*.csv") if f.is_file()]
        if flat_csvs and not processed_datasets:
            total_size = sum(f.stat().st_size for f in flat_csvs)
            labeled_count = 0
            latest_mtime = 0
            for csv_file in flat_csvs:
                try:
                    with open(csv_file, "r", encoding="utf-8") as f:
                        labeled_count += sum(1 for _ in f) - 1
                    mtime = csv_file.stat().st_mtime
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                except Exception:
                    pass

            processed_datasets.append({
                "id": "default",
                "name": "기본 라벨링 데이터셋",
                "processed_at": datetime.fromtimestamp(latest_mtime).isoformat() if latest_mtime else None,
                "labeled_count": labeled_count,
                "csv_path": str(labeled_dir),
                "size_mb": round(total_size / (1024 * 1024), 2),
                "status": "ready" if flat_csvs else "empty",
            })

        # 최신순 정렬
        processed_datasets.sort(key=lambda x: x.get("processed_at") or "", reverse=True)

    return {
        "stages": stages,
        "meta": meta,
        "latest_split": latest_split,
        "processed_datasets": processed_datasets,
    }


@app.get("/api/mlops/experiments")
async def list_experiments_endpoint(limit: int = 50):
    return {"experiments": list_experiments(limit=limit)}


@app.get("/api/mlops/experiments/{run_id}")
async def get_experiment_endpoint(run_id: str):
    exp = get_experiment(run_id)
    if not exp:
        raise HTTPException(status_code=404, detail="실험을 찾을 수 없습니다")
    return exp


@app.delete("/api/mlops/experiments/{run_id}")
async def delete_experiment_endpoint(run_id: str):
    """
    실험(모델) 삭제.

    - 활성 모델은 삭제 불가
    - DB에서 experiment, metrics 삭제
    - 모델 파일 디렉토리 삭제
    """
    import shutil
    from config import MODEL_DIR

    # 활성 모델 확인
    active = get_active()
    if active and active.get("run_id") == run_id:
        raise HTTPException(
            status_code=400,
            detail="운영 중인 모델은 삭제할 수 없습니다. 먼저 다른 모델을 활성화하세요."
        )

    # 실험 존재 여부 확인
    exp = get_experiment(run_id)
    if not exp:
        raise HTTPException(status_code=404, detail="실험을 찾을 수 없습니다")

    # DB에서 삭제
    from mlops.db import get_conn, init_db
    init_db()
    with get_conn() as conn:
        # metrics 먼저 삭제 (외래키)
        conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
        # predictions 삭제
        conn.execute("DELETE FROM predictions WHERE run_id = ?", (run_id,))
        # deployments 삭제 (비활성 상태의 배포 이력)
        conn.execute("DELETE FROM deployments WHERE run_id = ?", (run_id,))
        # experiment 삭제
        conn.execute("DELETE FROM experiments WHERE run_id = ?", (run_id,))

    # 모델 파일 디렉토리 삭제
    model_dir = MODEL_DIR / run_id
    if model_dir.exists():
        try:
            shutil.rmtree(model_dir)
            logger.info(f"모델 디렉토리 삭제: {model_dir}")
        except Exception as e:
            logger.warning(f"모델 디렉토리 삭제 실패 (무시): {e}")

    logger.info(f"실험 삭제 완료: {run_id}")
    return {
        "success": True,
        "message": "모델이 삭제되었습니다",
        "deleted_run_id": run_id,
    }


@app.get("/api/mlops/models/active")
async def get_active_model_endpoint():
    active = get_active()
    if not active:
        raise HTTPException(status_code=404, detail="활성 모델이 없습니다")
    return active


@app.post("/api/mlops/train")
async def trigger_training(req: TrainRequest):
    """
    비동기 학습 트리거. subprocess로 train.py 실행.
    응답은 즉시 반환되며 학습은 백그라운드 진행.
    완료되면 /api/mlops/experiments에서 새 run_id 조회 가능.
    """
    import subprocess
    import time
    import uuid as uuid_mod
    from pathlib import Path
    from config import BASE_DIR

    run_id = req.run_id or f"v_demo_{time.strftime('%Y%m%d_%H%M%S')}_{uuid_mod.uuid4().hex[:6]}"
    input_dir = req.input_dir or str(BASE_DIR / "data" / "labeled")

    # 분할 비율 검증
    if not (0 < req.train_ratio < 1 and 0 <= req.val_ratio < 1
            and req.train_ratio + req.val_ratio < 1):
        raise HTTPException(
            status_code=400,
            detail=(
                f"잘못된 분할 비율: train={req.train_ratio}, val={req.val_ratio} "
                "(0<train<1, 0<=val<1, train+val<1)"
            ),
        )

    # 모델 타입 검증
    valid_model_types = ["hist_gradient", "random_forest", "xgboost"]
    if req.model_type not in valid_model_types:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 모델 타입: {req.model_type}. 가능: {valid_model_types}",
        )

    cmd = [
        "python3", "-m", "training.train",
        "--input-dir", input_dir,
        "--run-id", run_id,
        "--max-iter", str(req.max_iter),
        "--max-depth", str(req.max_depth),
        "--learning-rate", str(req.learning_rate),
        "--train-ratio", str(req.train_ratio),
        "--val-ratio", str(req.val_ratio),
        "--model-type", req.model_type,
    ]

    log_dir = BASE_DIR / "models" / "saved"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.train.log"

    try:
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
        )
    except Exception as e:
        return {"success": False, "error": f"학습 프로세스 시작 실패: {e}"}

    # 작업 등록 (jobs 목록 조회용)
    register_job(run_id, "train", proc.pid)

    logger.info(f"학습 트리거: run_id={run_id}, pid={proc.pid}")
    return {
        "success": True,
        "run_id": run_id,
        "pid": proc.pid,
        "log_path": str(log_path),
        "command": " ".join(cmd),
        "message": "학습 시작. 1~2분 후 /api/mlops/experiments 에서 확인.",
    }


@app.post("/api/mlops/datasets/upload")
async def upload_dataset_zip(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    auto_build: bool = Form(False),
    mock: bool = Form(False),
    limit: Optional[int] = Form(None),
):
    """
    DXF zip 파일 업로드 → 학과 서버 디스크에 압축 해제 → 데이터셋 등록.

    Form 파라미터:
      - file: zip 파일 (필수)
      - name: 데이터셋 별칭 (없으면 zip 파일명에서 추출)
      - auto_build: True 면 업로드 후 자동으로 build_training_dataset 트리거
      - mock: auto_build 시 vLLM mock 사용 여부
      - limit: auto_build 시 처리 개수 제한

    응답:
      {
        "success": true,
        "dataset_id": "uploaded_20260427_...",
        "dxf_dir": "/home/.../uploads/.../dxf",
        "dxf_count": 98,
        "size_mb": 123.4,
        "auto_build": {"job_id": ..., "pid": ...} | null
      }
    """
    import json
    import shutil
    import time
    import uuid as uuid_mod
    import zipfile
    from config import BASE_DIR, DATASET_META_PATH

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail=f"zip 파일만 업로드 가능합니다 (받음: {file.filename})",
        )

    # 1. 업로드 디렉토리 생성
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    short_id = uuid_mod.uuid4().hex[:6]
    dataset_id = f"uploaded_{timestamp}_{short_id}"
    upload_root = BASE_DIR / "uploads" / dataset_id
    upload_root.mkdir(parents=True, exist_ok=True)

    # 2. zip 임시 저장
    zip_path = upload_root / "_upload.zip"
    try:
        with zip_path.open("wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunk
                f.write(chunk)
    except Exception as e:
        shutil.rmtree(upload_root, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {e}")

    upload_size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info(f"업로드 받음: {file.filename}, {upload_size_mb:.1f}MB → {zip_path}")

    # 3. 압축 해제
    extract_dir = upload_root / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # 보안: 절대경로/상위경로 차단
            for member in zf.namelist():
                if member.startswith("/") or ".." in Path(member).parts:
                    raise HTTPException(
                        status_code=400,
                        detail=f"위험한 경로 포함: {member}",
                    )
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        shutil.rmtree(upload_root, ignore_errors=True)
        raise HTTPException(status_code=400, detail="유효하지 않은 zip 파일입니다")
    except Exception as e:
        shutil.rmtree(upload_root, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"압축 해제 실패: {e}")

    # 4. DXF 파일 모으기 (zip 안 어디에 있든 dxf_dir 로 평탄화)
    dxf_files = list(extract_dir.rglob("*.dxf")) + list(extract_dir.rglob("*.DXF"))
    if not dxf_files:
        shutil.rmtree(upload_root, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail="zip 안에 .dxf 파일이 없습니다",
        )

    dxf_dir = upload_root / "dxf"
    dxf_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for src in dxf_files:
        dst = dxf_dir / src.name
        # 이름 충돌 시 _1, _2 추가
        idx = 1
        while dst.exists():
            dst = dxf_dir / f"{src.stem}_{idx}{src.suffix}"
            idx += 1
        shutil.move(str(src), str(dst))
        moved += 1

    # 정리: extracted/ 와 zip 은 삭제 (디스크 절약)
    shutil.rmtree(extract_dir, ignore_errors=True)
    try:
        zip_path.unlink()
    except Exception:
        pass

    total_size_mb = sum(f.stat().st_size for f in dxf_dir.glob("*.dxf")) / (1024 * 1024)

    # 5. 데이터셋 메타에 등록
    try:
        meta = {"datasets": []}
        if DATASET_META_PATH.exists():
            try:
                meta = json.loads(DATASET_META_PATH.read_text(encoding="utf-8"))
            except Exception:
                meta = {"datasets": []}
        if "datasets" not in meta or not isinstance(meta["datasets"], list):
            meta["datasets"] = []

        meta["datasets"].append({
            "id": dataset_id,
            "name": name or Path(file.filename).stem,
            "source": "upload",
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "uploaded_filename": file.filename,
            "dxf_dir": str(dxf_dir),
            "dxf_count": moved,
            "size_mb": round(total_size_mb, 2),
        })
        DATASET_META_PATH.parent.mkdir(parents=True, exist_ok=True)
        DATASET_META_PATH.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"데이터셋 메타 기록 실패 (계속 진행): {e}")

    logger.info(f"업로드 완료: {dataset_id} ({moved}개 DXF, {total_size_mb:.1f}MB)")

    response: dict = {
        "success": True,
        "dataset_id": dataset_id,
        "dxf_dir": str(dxf_dir),
        "dxf_count": moved,
        "size_mb": round(total_size_mb, 2),
        "auto_build": None,
    }

    # 6. (옵션) 업로드 직후 자동 빌드 트리거
    if auto_build:
        import subprocess
        job_id = f"build_{timestamp}_{short_id}"
        cmd = [
            "python3", "-m", "dataset.build_training_dataset",
            "--dxf-dir", str(dxf_dir),
            "--job-id", job_id,
        ]
        if mock:
            cmd.append("--mock")
        if limit:
            cmd.extend(["--limit", str(limit)])

        log_dir = BASE_DIR / "data" / "reports"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{job_id}.build.log"

        try:
            log_file = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR),
            )
            # 작업 등록 (jobs 목록 조회용)
            register_job(job_id, "build", proc.pid)
            response["auto_build"] = {
                "job_id": job_id,
                "pid": proc.pid,
                "log_path": str(log_path),
                "command": " ".join(cmd),
            }
            logger.info(f"자동 빌드 트리거: job_id={job_id}, pid={proc.pid}")
        except Exception as e:
            response["auto_build"] = {"error": f"자동 빌드 시작 실패: {e}"}

    return response


@app.post("/api/mlops/datasets/build")
async def trigger_dataset_build(req: CollectRequest):
    """
    비동기 데이터셋 재수집 트리거. build_training_dataset.py 실행.
    parse → render → vLLM Vision (또는 mock) → crop → label.
    완료되면 /api/mlops/datasets에서 단계별 카운트 갱신.
    """
    import subprocess
    import time
    import uuid as uuid_mod
    from config import BASE_DIR, EXTERNAL_DATASET_DIR

    dxf_dir = req.dxf_dir or str(EXTERNAL_DATASET_DIR)
    job_id = f"build_{time.strftime('%Y%m%d_%H%M%S')}_{uuid_mod.uuid4().hex[:6]}"

    cmd = [
        "python3", "-m", "dataset.build_training_dataset",
        "--dxf-dir", dxf_dir,
        "--job-id", job_id,
    ]
    if req.mock:
        cmd.append("--mock")
    if req.limit:
        cmd.extend(["--limit", str(req.limit)])

    log_dir = BASE_DIR / "data" / "reports"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id}.build.log"

    try:
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
        )
    except Exception as e:
        return {"success": False, "error": f"빌드 프로세스 시작 실패: {e}"}

    # 작업 등록 (jobs 목록 조회용)
    register_job(job_id, "build", proc.pid)

    logger.info(f"데이터셋 빌드 트리거: job_id={job_id}, pid={proc.pid}, dxf_dir={dxf_dir}")
    return {
        "success": True,
        "job_id": job_id,
        "pid": proc.pid,
        "log_path": str(log_path),
        "command": " ".join(cmd),
        "message": "데이터셋 빌드 시작. 5~15분 후 /api/mlops/datasets 에서 단계별 카운트 갱신 확인.",
    }


@app.get("/api/mlops/jobs/{job_id}/log")
async def get_job_log(job_id: str, tail: int = 100):
    """학습/빌드 로그 조회 (마지막 N줄)."""
    from config import BASE_DIR
    candidates = [
        BASE_DIR / "models" / "saved" / f"{job_id}.train.log",
        BASE_DIR / "data" / "reports" / f"{job_id}.build.log",
    ]
    for path in candidates:
        if path.exists():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
                return {
                    "job_id": job_id,
                    "log_path": str(path),
                    "tail": lines[-tail:],
                    "total_lines": len(lines),
                }
            except Exception as e:
                return {"error": str(e)}
    raise HTTPException(status_code=404, detail=f"로그 없음: {job_id}")


# ─── 모델 다운로드/업로드 ──────────────────────────────────
@app.get("/api/mlops/models/{run_id}/download")
async def download_model(run_id: str):
    """
    학습된 모델 파일 다운로드.

    모델 디렉토리에서 model.joblib 또는 model.pkl 파일을 찾아 반환.
    """
    from config import MODEL_DIR

    model_dir = MODEL_DIR / run_id
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"모델 디렉토리 없음: {run_id}")

    # joblib 또는 pkl 파일 찾기
    model_path = None
    for ext in [".joblib", ".pkl"]:
        candidate = model_dir / f"model{ext}"
        if candidate.exists():
            model_path = candidate
            break

    if not model_path:
        raise HTTPException(status_code=404, detail=f"모델 파일 없음: {run_id}")

    filename = f"model_{run_id}{model_path.suffix}"
    return FileResponse(
        path=str(model_path),
        media_type="application/octet-stream",
        filename=filename,
    )


@app.post("/api/mlops/models/upload")
async def upload_model(
    model_file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    algorithm: str = Form("Custom"),
):
    """
    외부에서 학습된 모델 파일 업로드 → 등록.

    Form 파라미터:
      - model_file: 모델 파일 (필수, .joblib/.pkl)
      - model_name: 모델 이름/버전 (선택, 기본값: 파일명)
      - algorithm: 알고리즘 종류 (선택, 기본값: "Custom")
    """
    import shutil
    import time
    import uuid as uuid_mod

    from config import MODEL_DIR

    # run_id 생성
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    short_id = uuid_mod.uuid4().hex[:6]
    run_id = f"uploaded_{timestamp}_{short_id}"

    # 저장 디렉토리 생성
    model_dir = MODEL_DIR / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # 파일 확장자 확인
    if not model_file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다")

    file_ext = Path(model_file.filename).suffix.lower()
    if file_ext not in [".joblib", ".pkl"]:
        file_ext = ".joblib"  # 기본값

    # 파일 저장
    model_path = model_dir / f"model{file_ext}"
    try:
        with model_path.open("wb") as f:
            shutil.copyfileobj(model_file.file, f)
    except Exception as e:
        shutil.rmtree(model_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {e}")

    # experiments 테이블에 등록
    from mlops.registry import register_experiment

    model_version = model_name or Path(model_file.filename).stem
    try:
        register_experiment(
            run_id=run_id,
            model_type=algorithm,
            status="completed",
            train_info={"source": "upload", "original_filename": model_file.filename},
        )
    except Exception as e:
        logger.warning(f"experiment 등록 실패 (계속 진행): {e}")

    logger.info(f"모델 업로드 완료: run_id={run_id}, path={model_path}")
    return {
        "success": True,
        "run_id": run_id,
        "model_version": model_version,
        "message": "모델이 성공적으로 등록되었습니다",
    }


# ─── 데이터셋 삭제 ─────────────────────────────────────────
@app.delete("/api/mlops/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """
    등록된 데이터셋 삭제.

    dataset_meta.json에서 해당 데이터셋 제거.
    실제 파일은 삭제하지 않음 (안전).
    """
    import json

    from config import DATASET_META_PATH

    if not DATASET_META_PATH.exists():
        raise HTTPException(status_code=404, detail="데이터셋 메타 파일 없음")

    try:
        meta = json.loads(DATASET_META_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"메타 파일 읽기 실패: {e}")

    datasets = meta.get("datasets", [])

    # 해당 데이터셋 찾기
    target = None
    for ds in datasets:
        if ds.get("id") == dataset_id:
            target = ds
            break

    if not target:
        raise HTTPException(status_code=404, detail=f"데이터셋을 찾을 수 없습니다: {dataset_id}")

    # 목록에서 제거
    datasets = [ds for ds in datasets if ds.get("id") != dataset_id]
    meta["datasets"] = datasets

    # 저장
    try:
        DATASET_META_PATH.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"메타 파일 저장 실패: {e}")

    logger.info(f"데이터셋 삭제: {dataset_id}")
    return {
        "success": True,
        "message": "데이터셋이 삭제되었습니다",
        "deleted_id": dataset_id,
    }


# ─── 작업 목록 조회 ────────────────────────────────────────
# 메모리에 작업 상태 저장 (서버 재시작 시 초기화됨)
ACTIVE_JOBS: dict = {}


def register_job(job_id: str, job_type: str, pid: int) -> None:
    """작업 등록 (내부 함수)."""
    import time
    ACTIVE_JOBS[job_id] = {
        "type": job_type,
        "pid": pid,
        "status": "running",
        "progress": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "completed_at": None,
        "message": "시작됨",
    }


def update_job_progress(job_id: str, progress: int, message: Optional[str] = None) -> None:
    """작업 진행률 업데이트 (내부 함수)."""
    if job_id in ACTIVE_JOBS:
        ACTIVE_JOBS[job_id]["progress"] = progress
        if message:
            ACTIVE_JOBS[job_id]["message"] = message


@app.get("/api/mlops/jobs")
async def list_jobs():
    """
    진행 중인 작업과 최근 완료된 작업 목록 반환.

    로그 파일 기반으로 작업 목록 조회.
    """
    import json
    import time

    from config import BASE_DIR

    jobs = []
    jobs_to_remove = []

    # 진행률 파일에서 실제 진행 상황 읽기
    def read_progress_file(job_id: str, job_type: str) -> tuple:
        """진행률 파일에서 progress와 message 읽기."""
        # 작업 유형에 따라 진행률 파일 위치가 다름
        if job_type == "build":
            progress_file = BASE_DIR / "data" / "reports" / f"{job_id}.progress"
        else:  # train
            progress_file = BASE_DIR / "models" / "saved" / f"{job_id}.progress"

        if progress_file.exists():
            try:
                data = json.loads(progress_file.read_text(encoding="utf-8"))
                return data.get("progress", 0), data.get("message", "처리 중...")
            except Exception:
                pass
        return 0, "시작 중..."

    # 메모리에 있는 작업 (실행 중)
    try:
        import psutil
        from datetime import datetime

        def is_job_process_running(pid: int, job_type: str) -> bool:
            """PID가 실제로 해당 잡 프로세스인지 확인 (PID 재사용 방지)."""
            if not pid or not psutil.pid_exists(pid):
                return False
            try:
                proc = psutil.Process(pid)
                cmdline = " ".join(proc.cmdline())
                # train 잡은 "training.train", build 잡은 "build_training_dataset" 포함
                if job_type == "train":
                    return "training.train" in cmdline
                elif job_type == "build":
                    return "build_training_dataset" in cmdline
                return False
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return False

        for job_id, job_info in list(ACTIVE_JOBS.items()):
            pid = job_info.get("pid")
            job_type = job_info.get("type", "")
            is_running = is_job_process_running(pid, job_type)

            # 진행률 파일이 100%면 프로세스 상태와 무관하게 완료 처리
            file_progress, file_message = read_progress_file(job_id, job_type)
            if file_progress >= 100 and job_info.get("status") == "running":
                is_running = False  # 강제 완료 처리

            if not is_running and job_info.get("status") == "running":
                # 프로세스가 종료됨 → 진행률 파일에서 최종 상태 읽고 completed로 변경
                final_progress, final_message = read_progress_file(job_id, job_info.get("type", ""))
                job_info["status"] = "completed"
                job_info["progress"] = final_progress if final_progress > 0 else 100
                job_info["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                job_info["message"] = final_message if final_progress > 0 else "완료됨"

            # 완료된 작업은 10분(600초) 후 메모리에서 제거
            # (최근 작업 목록에는 로그 파일 기반으로 계속 표시됨)
            if job_info.get("status") == "completed" and job_info.get("completed_at"):
                try:
                    completed_time = datetime.strptime(job_info["completed_at"], "%Y-%m-%dT%H:%M:%S")
                    elapsed = (datetime.now() - completed_time).total_seconds()
                    if elapsed > 600:  # 10분 후 메모리에서 제거
                        jobs_to_remove.append(job_id)
                        continue
                except (ValueError, TypeError):
                    pass

            # 진행률 파일에서 실제 진행 상황 읽기 (이미 위에서 읽음)
            progress = job_info.get("progress", 0)
            message = job_info.get("message", "처리 중...")
            if file_progress > 0:
                progress = file_progress
                message = file_message
                job_info["progress"] = progress
                job_info["message"] = message

            jobs.append({
                "job_id": job_id,
                "type": job_info.get("type"),
                "status": job_info.get("status", "unknown"),
                "progress": progress,
                "started_at": job_info.get("started_at"),
                "completed_at": job_info.get("completed_at"),
                "message": message,
            })

        # 오래된 완료 작업 메모리에서 제거
        for job_id in jobs_to_remove:
            del ACTIVE_JOBS[job_id]

    except ImportError:
        # psutil 없으면 메모리 작업만 반환
        for job_id, job_info in ACTIVE_JOBS.items():
            jobs.append({
                "job_id": job_id,
                "type": job_info.get("type"),
                "status": job_info.get("status", "unknown"),
                "progress": job_info.get("progress"),
                "started_at": job_info.get("started_at"),
                "completed_at": job_info.get("completed_at"),
                "message": job_info.get("message"),
            })

    # 로그 파일에서 최근 작업 추가 (메모리에 없는 것)
    log_dirs = [
        (BASE_DIR / "models" / "saved", "train"),
        (BASE_DIR / "data" / "reports", "build"),
    ]

    existing_ids = {j["job_id"] for j in jobs}

    for log_dir, job_type in log_dirs:
        if not log_dir.exists():
            continue
        for log_file in sorted(log_dir.glob(f"*.{job_type}.log"), reverse=True)[:10]:
            job_id = log_file.stem.replace(f".{job_type}", "")
            if job_id in existing_ids:
                continue

            stat = log_file.stat()
            jobs.append({
                "job_id": job_id,
                "type": job_type,
                "status": "completed",
                "progress": 100,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_ctime)),
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
                "message": f"로그 파일: {log_file.name}",
            })
            existing_ids.add(job_id)

    # 최근 작업 순 정렬
    jobs.sort(key=lambda x: x.get("started_at", ""), reverse=True)

    return {"jobs": jobs[:20]}


# ─── 오분류 수정 (Corrections) API ────────────────────────────
class CorrectionRequest(BaseModel):
    file_id: Optional[str] = None
    entity_id: Optional[str] = None
    raw_layer: Optional[str] = None
    original_class: str
    corrected_class: str
    confidence: Optional[float] = None


@app.post("/api/corrections")
async def submit_correction(req: CorrectionRequest):
    """
    사용자가 잘못된 분류를 수정할 때 기록.

    예: wall로 분류되었지만 실제로는 door인 경우.
    """
    from mlops.db import get_conn, init_db

    init_db()

    # 현재 활성 모델의 run_id 가져오기
    active = get_active()
    run_id = active.get("run_id") if active else None

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO corrections (run_id, file_id, entity_id, raw_layer, original_class, corrected_class, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                req.file_id,
                req.entity_id,
                req.raw_layer,
                req.original_class,
                req.corrected_class,
                req.confidence,
            ),
        )

    logger.info(f"오분류 수정 기록: {req.original_class} → {req.corrected_class}")
    return {
        "success": True,
        "message": "수정 내용이 기록되었습니다",
        "original_class": req.original_class,
        "corrected_class": req.corrected_class,
    }


@app.get("/api/mlops/stats")
async def get_mlops_stats():
    """
    MLOps 통계 반환: 총 분류 횟수, 오분류 횟수(테스트셋 기반), 평균 신뢰도 등.
    """
    import json as json_lib
    from mlops.db import get_conn, init_db

    init_db()

    stats = {
        "total_predictions": 0,
        "total_test_samples": 0,
        "test_misclassifications": 0,
        "test_accuracy": None,
        "misclassification_rate": 0.0,
        "average_confidence": None,
        "confusion_matrix": None,
    }

    with get_conn() as conn:
        # 총 예측 수 (운영 중 샘플링된 예측)
        row = conn.execute("SELECT COUNT(*) as cnt FROM predictions").fetchone()
        stats["total_predictions"] = row["cnt"] if row else 0

        # 평균 신뢰도
        row = conn.execute("SELECT AVG(confidence) as avg_conf FROM predictions WHERE confidence IS NOT NULL").fetchone()
        if row and row["avg_conf"]:
            stats["average_confidence"] = round(row["avg_conf"], 4)

        # 현재 활성 모델의 테스트셋 메트릭에서 오분류 계산
        active = get_active()
        if active:
            run_id = active.get("run_id")
            # test split의 메트릭 조회
            row = conn.execute(
                """
                SELECT accuracy, confusion_matrix, n_samples
                FROM metrics
                WHERE run_id = ? AND split = 'test'
                """,
                (run_id,)
            ).fetchone()

            if row:
                stats["test_accuracy"] = row["accuracy"]
                stats["total_test_samples"] = row["n_samples"] or 0

                # confusion matrix에서 오분류 수 계산
                if row["confusion_matrix"]:
                    try:
                        cm = json_lib.loads(row["confusion_matrix"])
                        # confusion matrix: 대각선 = 정분류, 나머지 = 오분류
                        total = 0
                        correct = 0
                        for i, row_vals in enumerate(cm):
                            for j, val in enumerate(row_vals):
                                total += val
                                if i == j:
                                    correct += val
                        stats["test_misclassifications"] = total - correct
                        stats["confusion_matrix"] = cm
                        if total > 0:
                            stats["misclassification_rate"] = round((total - correct) / total * 100, 2)
                    except (json_lib.JSONDecodeError, TypeError):
                        pass

    return stats


@app.post("/api/mlops/deploy")
async def deploy_endpoint(req: DeployRequest):
    # run_id 존재 여부 확인
    exp = get_experiment(req.run_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"run_id not found: {req.run_id}")

    set_active(req.run_id, environment=req.environment, notes=req.notes or "")

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
