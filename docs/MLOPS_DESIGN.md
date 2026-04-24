# MLOps 설계안

> 작성일: 2026-04-23 (초안, 호민님 리뷰 대기)
> 대상: ai_layer_classifier — 학과 AI 서버 배포용

## 1. 목표

학습된 분류 모델을 **한 번 학습하고 끝**이 아니라, 지속적으로 성능을 측정하고 재학습/교체/롤백할 수 있는 운영 체계를 구축한다.

요구사항:
1. **성능 확인** — 학습 직후 + 운영 중 지속 모니터링
2. **재수집/재학습** — 기준 충족 시 자동/수동 재학습
3. **모델 비교** — 버전 간 metric side-by-side
4. **학과 AI 서버 배포** — FastAPI로 단일 서비스 형태

## 2. 전체 아키텍처

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ai_layer_classifier (port 8001)                    │
│                                                                        │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────────────────┐  │
│  │ /api/classify │  │ /api/mlops/*  │  │  Pipeline Runner         │  │
│  │ 추론 엔드포인트│  │ 운영 엔드포인트│  │  (train / retrain 트리거)│  │
│  └───────┬───────┘  └───────┬───────┘  └──────────┬───────────────┘  │
│          │                  │                     │                   │
│          ▼                  ▼                     ▼                   │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                    mlops 코어 모듈                              │   │
│  │  registry.py   │  tracker.py  │  monitor.py  │  trigger.py    │   │
│  │  (model/dataset│  (experiment │  (prediction │  (retrain rules)│  │
│  │   버전 관리)    │   기록)       │   로그)       │                 │   │
│  └───────────────────┬───────────────────────────────────────────┘   │
│                      │                                                 │
│  ┌───────────────────▼───────────────────────────────────────────┐   │
│  │                  SQLite: mlops.db                              │   │
│  │  experiments | metrics | predictions | datasets | deployments │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                 파일 저장소                                    │   │
│  │  models/saved/{run_id}/  (pkl, config.json, metrics.json)     │   │
│  │  data/splits/{dataset_version}/  (train/val/test CSV)         │   │
│  │  data/golden_test/  (고정 평가셋 — 모든 모델이 동일하게 평가)   │   │
│  └────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
              │                                          ▲
              │ HTTP (OpenAI 호환)                       │ /api/classify
              ▼                                          │
     ┌──────────────────────┐                 ┌─────────────────────────┐
     │ 학과 vLLM 프록시       │                 │  building_cesium        │
     │ cellm.gachon.ac.kr    │                 │  backend (port 8000)    │
     │  text/vision/embed    │                 └─────────────────────────┘
     └──────────────────────┘
```

## 3. 데이터 모델 (SQLite: `mlops.db`)

### 3.1 `datasets` — 데이터셋 버전

```sql
CREATE TABLE datasets (
  id TEXT PRIMARY KEY,                -- "dataset_v1", "dataset_v2"
  name TEXT NOT NULL,
  source TEXT,                        -- "dataset1_crawled", "user_upload"
  file_count INTEGER,
  total_entities INTEGER,
  class_distribution TEXT,            -- JSON: {"wall": 36669, ...}
  split_dir TEXT,                     -- data/splits/dataset_v1/
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  is_active INTEGER DEFAULT 0         -- 현재 학습 기본 데이터셋 여부
);
```

### 3.2 `experiments` — 학습 run 1회

```sql
CREATE TABLE experiments (
  run_id TEXT PRIMARY KEY,            -- UUID
  dataset_id TEXT NOT NULL,
  model_type TEXT NOT NULL,           -- "xgboost", "random_forest"
  hyperparams TEXT NOT NULL,          -- JSON
  features_used TEXT,                 -- JSON: {"embedding": true, "geometry": true, ...}
  model_path TEXT,                    -- models/saved/{run_id}/model.pkl
  training_time_seconds REAL,
  status TEXT NOT NULL,               -- "running" | "completed" | "failed"
  error_message TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);
```

### 3.3 `metrics` — 실험별 성능 지표

```sql
CREATE TABLE metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  split TEXT NOT NULL,                -- "val" | "test" | "golden"
  overall_accuracy REAL,
  overall_f1_macro REAL,
  overall_f1_weighted REAL,
  per_class_precision TEXT,           -- JSON: {"wall": 0.93, ...}
  per_class_recall TEXT,
  per_class_f1 TEXT,
  confusion_matrix TEXT,              -- JSON 2D array
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (run_id) REFERENCES experiments(run_id)
);
```

### 3.4 `predictions` — 추론 로그 (샘플링해서 기록)

```sql
CREATE TABLE predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT,                        -- 어느 모델이 예측했는지
  file_id TEXT,
  entity_id TEXT,
  raw_layer TEXT,
  entity_type TEXT,
  predicted_class TEXT,
  confidence REAL,
  is_new_layer INTEGER,               -- 학습 시 못본 레이어명?
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

전체 로그는 양이 많으니 **샘플링(기본 10%) 저장**. 집계용.

### 3.5 `deployments` — 활성 모델 지정

```sql
CREATE TABLE deployments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  environment TEXT DEFAULT 'production',  -- "production" | "shadow" | "canary"
  deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  deployed_by TEXT,                    -- 사용자 식별
  is_active INTEGER DEFAULT 1,
  notes TEXT,
  FOREIGN KEY (run_id) REFERENCES experiments(run_id)
);
```

운영: environment별로 is_active=1인 row는 최대 1개. 교체 시 이전 row는 is_active=0으로 갱신.

## 4. 파일 레이아웃

```
ai_layer_classifier/
├── mlops.db                                  ← SQLite (코드 밖, 데이터)
├── models/saved/
│   ├── {run_id}/
│   │   ├── model.pkl                         ← XGBoost/sklearn 모델
│   │   ├── feature_pipeline.pkl              ← embedding 캐시/scaler
│   │   ├── config.json                       ← 학습 시 하이퍼파라미터
│   │   └── metrics.json                      ← 전체 metric
│   └── registry.json                         ← 전체 run 인덱스 (DB 미러)
├── data/
│   ├── splits/{dataset_version}/             ← train/val/test CSV
│   └── golden_test/
│       ├── entities.csv                      ← 수동 라벨링된 5~10개 DXF
│       └── README.md                         ← 어떤 파일이고 왜 선정했는지
└── mlops/                                    ← MLOps 코어 모듈 (신규)
    ├── __init__.py
    ├── registry.py                           ← 모델/데이터셋 CRUD
    ├── tracker.py                            ← experiments 기록
    ├── monitor.py                            ← predictions 로그 + 집계
    ├── trigger.py                            ← 재학습 규칙 판정
    └── db.py                                 ← SQLite 연결/마이그레이션
```

## 5. 재학습 트리거 규칙

네 가지 규칙 중 **하나라도** 맞으면 재학습 "제안" 상태로. 실제 학습은 수동 승인(또는 전체 자동 모드 플래그).

| 규칙 | 조건 | 근거 |
|---|---|---|
| R1. 데이터 증분 | 누적 신규 DXF ≥ 현재 학습셋의 20% | 충분한 추가 데이터 |
| R2. Confidence drift | 최근 7일 avg confidence < 학습 시 avg × 0.9 | 모델이 헷갈려 함 |
| R3. 신규 레이어명 | 최근 7일 예측 중 학습 시 못 본 레이어명 비율 ≥ 30% | 데이터 분포 변화 |
| R4. 수동 | 관리자가 `POST /api/mlops/retrain` 호출 | 명시적 요청 |

`mlops/trigger.py`의 `check_retrain_needed()` 함수가 매 N시간(기본 24h) 마다, 또는 API 호출 시 판정.

## 6. 모델 비교

두 가지 비교 모드:

**6.1 Golden test 기반** — 모든 모델을 **같은 holdout 평가셋**에 돌려서 비교:
- `data/golden_test/entities.csv` = 수동 라벨링된 절대 기준 (학습에 절대 미사용)
- 새 모델 학습 완료 직후 자동 평가
- DB에 `metrics` 저장 (split='golden')
- API: `GET /api/mlops/compare?a={run_id_A}&b={run_id_B}&split=golden`

**6.2 Shadow inference** — 새 모델을 **active 바꾸지 않고 병렬로 돌려서** 실제 트래픽에서 성능 비교:
- `/api/classify` 호출 시 active + shadow 모두 추론
- 응답은 active 기준, 결과 diff만 DB에 저장
- 일정 기간 후 safe 판정되면 승격

## 7. API 엔드포인트

```
# 기존 (building_cesium 연동)
POST /api/classify                    # 실제 분류 서빙 (active 모델)
GET  /health

# MLOps 운영
GET  /api/mlops/datasets              # 데이터셋 목록
POST /api/mlops/datasets              # 새 데이터셋 등록 (DXF 추가 등)

GET  /api/mlops/experiments           # 전체 학습 run
GET  /api/mlops/experiments/{run_id}  # 특정 run 상세
POST /api/mlops/experiments           # 학습 시작 (async)
     body: { dataset_id, model_type, hyperparams }

GET  /api/mlops/metrics/{run_id}      # 모델 metric
GET  /api/mlops/compare               # 모델 비교 (?a=..&b=..&split=..)

GET  /api/mlops/deployments/active    # 현재 active 모델
POST /api/mlops/deploy                # 모델 승격 ({run_id, env})
POST /api/mlops/rollback              # 직전 모델로

GET  /api/mlops/monitor/recent        # 최근 추론 성능 요약
GET  /api/mlops/retrain-check         # 재학습 필요 여부 + 규칙별 상태
POST /api/mlops/retrain               # 재학습 트리거 (수동)
```

모든 쓰기 API는 **admin 인증** 필요 (building_cesium의 admin_accounts 스키마 재사용 가능).

## 8. 학습/재학습 워크플로우

```
[1] 새 DXF 수집 (user upload 또는 crawl)
         ↓
[2] parse_dxf → render_preview → detect_floorplan → crop
         ↓
[3] build_dataset: weak label 부착 → data/splits/dataset_v{N}/ 에 저장
         ↓
[4] mlops/trigger.py: 재학습 규칙 체크
         ├─ 조건 안 맞음 → 대기
         └─ 조건 맞음 → 재학습 제안 상태 기록
         ↓ (관리자 승인 또는 auto-mode)
[5] training/train.py 실행 (async, background task)
         ├─ run_id 생성, status='running'
         ├─ XGBoost 학습 → model.pkl 저장
         ├─ val/test/golden 평가 → metrics 저장
         └─ status='completed'
         ↓
[6] 자동 비교: 새 run vs 현재 active
         ├─ f1_macro 개선 ≥ 1%p AND golden regression 없음 → 승격 제안
         └─ 그 외 → "보류" 상태
         ↓
[7] POST /api/mlops/deploy {run_id}
         ├─ deployments 테이블 업데이트
         └─ runtime의 active 모델 교체 (메모리 swap)
         ↓
[8] 계속 /api/classify 수신 → predictions 샘플링 기록
         → monitor.py가 집계 → (1)로 루프
```

## 9. Dashboard (선택 — v2)

building_cesium의 admin 콘솔에 AI 페이지가 이미 있음 (`frontend/app/admin/ai/page.tsx`). 이걸 확장해서:
- 실험 이력 테이블
- 현재 active 모델 + 최근 metric
- 재학습 제안 배너
- 모델 간 비교 차트

v1에서는 API만 노출하고 곡선/차트는 v2에서. JSON 리스폰스로 curl/Postman 검증 가능하게.

## 10. 구현 우선순위

졸프 일정 고려해서 최소 필요분부터:

**P0 (반드시 — 심사/발표 시연에 필요)**
- `mlops/db.py` + SQLite 스키마
- `mlops/registry.py` (experiments/metrics CRUD)
- `training/train.py` 안에 tracker 연동
- `main.py`: `/api/classify`, `/api/mlops/experiments`, `/api/mlops/deploy`, `/api/mlops/compare`
- golden_test 5개 수동 라벨링

**P1 (시연 가능하면 좋음)**
- `mlops/monitor.py` 추론 로깅
- `mlops/trigger.py` 규칙 판정
- `/api/mlops/retrain-check`
- building_cesium admin 페이지 연결

**P2 (발표 후)**
- shadow inference
- 자동 승격
- 실제 dashboard UI

## 11. 학과 서버 배포

Dockerfile 하나로 packaging:
```
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

학과 서버가 어떤 환경인지(IP, GPU 유무, 학과 내부망 제한 등) 신재훈 팀장에게 확인 필요.

`building_cesium/backend/main.py:873`에서 이미 `AI_SERVER_URL` 환경변수로 프록시 주소 지정 가능하므로, 학과 서버 IP가 확정되면 환경변수만 바꾸면 됨.

## 12. 리스크 + 완화책

| 리스크 | 영향 | 완화 |
|---|---|---|
| 학과 서버 GPU 없음 | 임베딩 생성 느림 | 배치 처리 + 캐시, CPU도 nomic-embed는 허용 가능 수준 |
| SQLite 동시쓰기 경합 | 멀티워커 시 락 | WAL 모드 + writer 1개로 제한 (v1 단일 워커) |
| golden test 5개 너무 적음 | 평가 신뢰도 낮음 | 추후 확장, 현재는 per-class 최소 20 samples 목표 |
| 학습 중 /classify 다운 | SLA 깨짐 | 학습은 background, active 모델은 별도 로딩 |
| 임베딩 API 쿼터 초과 | 학습/추론 실패 | 로컬 sentence-transformer 폴백 준비 |
