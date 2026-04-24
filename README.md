# Team 건치 — AI Layer Classifier

> **CAD 도면(DXF) 레이어 자동 분류 모델 + MLOps 서빙 인프라**
> 2026 종합설계프로젝트 (Team 건치 / Gachon CS)

가천대학교 컴퓨터공학과 종합설계 프로젝트 "Team 건치(Geonchi)"의 AI 모듈.
CAD 도면에서 **레이어 이름이 달라도** wall / door / window 를 자동 분류하여,
메인 서비스 [building_cesium](https://github.com/LeeHome2/Team_Gunchi)이 3D 매스 생성에 사용할 레이어를 자동 선별합니다.

---

## 목차

- [배경 및 목표](#배경-및-목표)
- [아키텍처](#아키텍처)
- [기술 스택](#기술-스택)
- [디렉토리 구조](#디렉토리-구조)
- [빠른 시작](#빠른-시작)
- [파이프라인 상세](#파이프라인-상세)
- [API 문서](#api-문서)
- [MLOps 설계](#mlops-설계)
- [성능 및 결과](#성능-및-결과)
- [한계 및 향후 계획](#한계-및-향후-계획)
- [팀 정보](#팀-정보)

---

## 배경 및 목표

메인 프로젝트 `building_cesium`은 개인 건축주가 업로드한 DXF 도면을 Cesium 3D 지도 위에 건물 매스로 시각화합니다.
이 과정에서 **어느 레이어가 벽체인지** 자동으로 판별해야 하는데, 실제 CAD 도면들은:

- 레이어 이름 표기가 파일마다 다름 (`WALL`, `Wall`, `A-WALL`, `Muro`, `I-WALL`, ...)
- 한 CAD 파일에 평면도 + 입면도 + 단면도가 섞여있는 경우가 많음
- 같은 레이어에 벽체와 가구가 혼재 (`Interior`, `Arch-plan` 등 mixed layer)

본 모듈은 이런 실데이터의 다양성에도 **정확히 wall/door/window를 자동 분류**하는 것을 목표로 합니다.

**핵심 설계 결정**:
- 레이어 이름만으로는 분류 불가 → **지오메트리 feature + 레이어명 char n-gram** 결합
- 멀티 도면 섞인 CAD → **학과 vLLM Vision 모델로 평면도 영역만 선별**
- 외벽/내벽 구분 없이 **4-class** (wall / door / window / other) — 매스 생성엔 충분
- 학습 없는 레이블 확보 → **Weak Supervision** (키워드 + 지오메트리 규칙)

---

## 아키텍처

### 시스템 전체

```
┌─────────────────────────────────────────────────────────────────┐
│                  AWS EC2 (building_cesium)                       │
│  ┌────────────────┐         ┌─────────────────────────────────┐ │
│  │ Next.js 프론트  │◄───────►│ FastAPI 백엔드                  │ │
│  │ (Cesium Viewer) │         │ - DXF 파싱 (ezdxf)              │ │
│  └────────────────┘         │ - PostgreSQL (프로젝트/유저)      │ │
│                              │ - /api/classify → AI 서버 프록시 │ │
│                              └──────────────┬──────────────────┘ │
└─────────────────────────────────────────────┼──────────────────┘
                                               │ HTTP (엔티티 JSON)
              ┌────────────────────────────────▼────────────────┐
              │   학과 AI 서버 (Rocky Linux, RTX A5000 x2)       │
              │  ┌──────────────────────────────────────────┐   │
              │  │ FastAPI (main.py, port 8001)             │   │
              │  │  - /api/classify (상시 추론)              │   │
              │  │  - /api/mlops/* (레지스트리/배포)          │   │
              │  └──────────────────────────────────────────┘   │
              │  ┌──────────────────────────────────────────┐   │
              │  │ SQLite (mlops.db)                        │   │
              │  │  experiments / metrics / deployments     │   │
              │  └──────────────────────────────────────────┘   │
              │  ┌──────────────────────────────────────────┐   │
              │  │ Models (joblib)                          │   │
              │  │  HistGradientBoostingClassifier + TF-IDF │   │
              │  └──────────────────────────────────────────┘   │
              └────────────────────────┬─────────────────────────┘
                                        │ OpenAI 호환 HTTP
              ┌────────────────────────▼─────────────────────────┐
              │  학과 vLLM 프록시 (cellm.gachon.ac.kr)            │
              │  Qwen3.5-35B (text/vision) + nomic-embed         │
              └──────────────────────────────────────────────────┘
```

### 학습 파이프라인 (전처리 + 학습)

```
  DXF 98개 (크롤링 데이터셋)
         │
         ├────────────────────┬──────────────────┐
         │                    │                  │
         ▼                    ▼                  │
  parse_dxf             render_preview           │
  (ezdxf)               (ezdxf+matplotlib)       │
  → *.csv               → *.png + extents.json   │
         │                    │                  │
         │                    ▼                  │
         │             detect_floorplan          │
         │             (vLLM Vision)             │
         │             → *.bboxes.json           │
         │                    │                  │
         └──┬─────────────────┘                  │
            ▼                                    │
       crop_entities_by_bbox                     │
       → 평면도 영역의 엔티티만 남긴 CSV           │
            │                                    │
            ▼                                    │
       weak_label  ◄──────────── 키워드 사전      │
       → wall / door / window / other            │
            │                                    │
            ▼                                    │
       train (HistGradientBoosting)              │
       → model.joblib + feature_pipeline.joblib  │
            │                                    │
            ▼                                    │
       mlops.db 등록 + /api/classify 서빙         │
```

### 추론 플로우 (런타임)

```
유저 DXF 업로드
   ↓
AWS building_cesium (ezdxf로 엔티티 JSON 변환)
   ↓ POST /api/classify { entities: [...] }
학과 AI 서버
   ├─ FeatureExtractor (TF-IDF + one-hot + 지오메트리)
   ├─ HistGradientBoosting predict
   └─ class_counts + layer_decisions 반환
   ↓
AWS가 wall 레이어만 선별 → GLB 매스 생성
   ↓
Cesium에서 3D 렌더
```

---

## 기술 스택

### 코어
| 라이브러리 | 버전 | 용도 |
|---|---|---|
| Python | 3.11+ (학과 서버는 3.9) | 런타임 |
| **ezdxf** | ≥1.1.0 | DXF 파싱, matplotlib 백엔드 렌더링 |
| numpy | ≥1.24 | 수치 연산 |
| pandas | ≥2.0 | 엔티티 DataFrame |

### 머신러닝
| 라이브러리 | 버전 | 용도 |
|---|---|---|
| **scikit-learn** | ≥1.3 | **HistGradientBoostingClassifier** + TF-IDF |
| joblib | ≥1.3 | 모델 직렬화 |

> XGBoost는 설계 검토했으나 학과 서버 환경을 고려해 sklearn 내장 HistGB로 대체.
> 동일한 Histogram-based GBDT 알고리즘, 성능 동급, 의존성 최소화.

### LLM / 외부 API
| 라이브러리 | 용도 |
|---|---|
| openai (≥1.40) | OpenAI 호환 SDK로 학과 vLLM 프록시 호출 |
| httpx | HTTP 클라이언트 |

**학과 vLLM 프록시** (`cellm.gachon.ac.kr:8000/v1`) — OpenAI 호환
- `vision` 모델 (Qwen3.5-35B multimodal): **평면도 bbox 검출**
- `text` 모델 (Qwen3.5-35B): 레이어명 분류 실험용
- `embedding` 모델 (nomic-embed-text-v1.5, 768차원): feature 확장 계획

### 렌더링
| 라이브러리 | 용도 |
|---|---|
| matplotlib | ezdxf 백엔드, DXF → PNG 렌더 |
| Pillow | 이미지 크기 확인 / 재압축 |

### 서빙
| 라이브러리 | 용도 |
|---|---|
| FastAPI (≥0.109) | REST API 서버 |
| uvicorn[standard] | ASGI 서버 |
| python-multipart | 파일 업로드 |

### 데이터 저장
- SQLite3 (Python 내장) — MLOps DB (`mlops.db`)
- 파일시스템 — 모델 바이너리, 중간 데이터

---

## 디렉토리 구조

```
ai_layer_classifier/
├── config.py                           # 전역 설정 (경로, LLM, 분류 클래스)
├── main.py                             # FastAPI 추론 서버
├── requirements.txt
├── README.md                           # 이 문서
├── QUICKSTART.md                       # 단계별 실행 가이드
│
├── dataset/
│   ├── parse_dxf.py                    # [1] DXF → 엔티티 CSV
│   ├── render_preview.py               # [2] DXF → PNG + extents 메타
│   ├── detect_floorplan.py             # [3] vLLM Vision → 평면도 bbox
│   ├── crop_entities_by_bbox.py        # [4] bbox로 엔티티 필터링
│   ├── weak_label.py                   # [5] 키워드 라벨링
│   └── build_training_dataset.py       # 1~5 전체 orchestrator
│
├── training/
│   ├── feature_extractor.py            # TF-IDF + one-hot + 지오메트리
│   └── train.py                        # [6] HistGB 학습 + MLOps 기록
│
├── inference/
│   └── predictor.py                    # 모델 로드 + 예측
│
├── mlops/
│   ├── db.py                           # SQLite 스키마 + WAL 모드
│   └── registry.py                     # experiments/deployments CRUD
│
├── llm/
│   ├── client.py                       # 학과 vLLM 래퍼 (vision/text/embedding)
│   └── parse_response.py               # JSON 응답 검증 + 다중 bbox 지원
│
├── tests/
│   ├── smoke_test_llm.py               # vLLM 3모델 연결 테스트
│   ├── test_layer_labeling_llm.py      # LLM 라벨링 정확도 실험
│   └── README.md                       # 테스트 실행 가이드
│
├── configs/
│   ├── dataset_meta.json
│   ├── train_params.json
│   └── llm_prompts/floorplan_bbox_prompt.txt
│
├── data/
│   ├── processed/      # parse_dxf 출력 (.csv + .meta.json + .bboxes.json)
│   ├── preview/        # render_preview 출력 (.png + .preview.json)
│   ├── cropped/        # crop 결과 (평면도별 개별 CSV)
│   ├── labeled/        # weak_label 출력
│   └── reports/        # 데이터셋 분석 / 빌드 리포트
│
├── models/saved/       # 학습된 모델들 (run_id별 폴더)
├── mlops.db            # SQLite 레지스트리 (런타임 생성)
│
└── docs/
    └── MLOPS_DESIGN.md                 # MLOps 상세 설계서
```

---

## 빠른 시작

상세 명령은 [QUICKSTART.md](QUICKSTART.md) 참고.

### 1. 설치

```bash
git clone https://github.com/LeeHome2/Team_Gunchi_classifier.git
cd Team_Gunchi_classifier

python -m venv venv
source venv/bin/activate           # Linux/Mac
# venv\Scripts\activate            # Windows

pip install -r requirements.txt
```

### 2. 환경변수 설정 (`.env`)

학과 vLLM API 키는 보안상 레포에 포함돼 있지 않음.
`.env.example`을 복사해 본인 키를 입력해야 함.

```bash
# 템플릿 복사
cp .env.example .env

# .env 파일을 열어서 LLM_API_KEY=sk-vllm-xxxxx... 입력
```

**팀 API 키 배포 경로**:
- Notion 팀 페이지 또는 내부 채널로 공유
- 저장소에는 절대 커밋하지 말 것 (`.gitignore`에 `.env` 포함)
- 외부 기여자는 본인 발급 키 사용

**환경변수 전체 목록** (선택 항목은 기본값 유지 가능):

| 변수 | 필수 | 기본값 | 설명 |
|---|---|---|---|
| `LLM_API_KEY` | ✅ | (없음) | 학과 vLLM 프록시 팀 키 (`sk-vllm-...`) |
| `LLM_BASE_URL` | 선택 | `http://cellm.gachon.ac.kr:8000/v1` | vLLM 엔드포인트 |
| `LLM_TEXT_MODEL` | 선택 | `text` | 텍스트 모델 별칭 |
| `LLM_VISION_MODEL` | 선택 | `vision` | 비전 모델 별칭 |
| `LLM_EMBEDDING_MODEL` | 선택 | `embedding` | 임베딩 모델 별칭 |
| `LLM_TIMEOUT` | 선택 | `60` | HTTP 타임아웃(초) |
| `EXTERNAL_DATASET_DIR` | 선택 | `../데이터셋1-dxf/dxf` | 학습용 원본 DXF 폴더 |
| `MLOPS_DB_PATH` | 선택 | `./mlops.db` | SQLite DB 경로 |
| `SERVING_HOST` | 선택 | `0.0.0.0` | FastAPI 바인딩 호스트 |
| `SERVING_PORT` | 선택 | `8001` | FastAPI 포트 |

`.env` 파일은 `config.py`가 import 시 자동으로 로드함 (python-dotenv).

### 3. 학과 vLLM 연결 확인

```bash
python -m tests.smoke_test_llm
```

`text / embedding / vision` 모두 PASS 확인.

### 3. 전체 파이프라인 (전처리 → 학습)

```bash
# 전처리: DXF → CSV → PNG → bbox → crop → label (약 15~20분, vLLM 호출 포함)
python -m dataset.build_training_dataset --dxf-dir ../데이터셋1-dxf/dxf

# 학습: HistGB (약 30초 ~ 2분)
python -m training.train --input-dir data/labeled
```

### 4. Active 모델 지정

```bash
python -c "from mlops.registry import set_active; set_active('<run_id>')"
```

### 5. FastAPI 서버 실행

```bash
python main.py
# → http://localhost:8001/docs 에서 Swagger UI 확인
```

### 6. 추론 테스트

```bash
curl -X POST http://localhost:8001/api/classify \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "test",
    "entities": [
      {"entity_id":"1","entity_type":"LINE","raw_layer":"WALL",
       "length":3.5,"bbox_width":3.5,"bbox_height":0.01,"aspect_ratio":350}
    ]
  }'
```

---

## 파이프라인 상세

### [1] DXF → CSV (parse_dxf)
ezdxf로 DXF 파일을 열어 모든 엔티티(LINE, LWPOLYLINE, ARC, TEXT, ...)를 행 단위 CSV로 추출.

**추출 필드**: `entity_id, entity_type, raw_layer, length, bbox_*, center_*, start/end, radius, n_vertices, text_content, is_geometric`

### [2] DXF → PNG (render_preview)
ezdxf의 matplotlib 백엔드로 PNG 렌더. 학과 vLLM 제약(2~5MB)을 맞춰 자동 다운샘플. CAD 좌표계의 extents를 JSON으로 저장 (bbox 역변환에 필수).

### [3] 평면도 검출 (detect_floorplan)
PNG를 학과 vLLM Vision 모델에 전송하여 평면도 영역의 **정규화 bbox(0~1)** 를 받음.
- 다중 평면도 지원 (`floorplans: [{label, bbox}, ...]`)
- 캐시 레이어 (파일당 1회 호출, 재실행 시 캐시 사용)
- `--mock` 모드 (개발 환경에서 vLLM 없이 구조 검증)

**프롬프트** (`configs/llm_prompts/floorplan_bbox_prompt.txt`):
- 평면도 특징 (벽체/방 구획/문 호/창 표시) 명시
- 단면도/입면도/3D/표 제외 규칙
- 좌표 정규화 (0~1) 강조
- JSON 응답 강제

### [4] 엔티티 크롭 (crop_entities_by_bbox)
정규화 bbox × CAD extents 로 역변환하여 CSV에서 해당 영역 엔티티만 필터.
검출 실패 시 **전체 이미지를 단일 평면도로 간주하는 fallback** (robustness).

### [5] Weak Label (weak_label)
4단계 필터:
1. 노이즈 레이어 제거 (`PDF_*`, `cadblocks`, `Defpoints`)
2. 비지오메트리 엔티티 (TEXT/DIMENSION/HATCH/INSERT) → other
3. 키워드 매칭 (wall/door/window 정규식)
4. 지오메트리 sanity check (도면 전체 크기와 유사한 외곽선 제외)

### [6] 학습 (train)
**FeatureExtractor** (learning/feature_extractor.py):
- **TF-IDF char n-gram** (2~4) on `raw_layer` — 다국어/표기 변이 흡수
- **Entity type one-hot**
- **지오메트리** (`log1p(length)`, bbox 크기, aspect_ratio)

**모델**: `HistGradientBoostingClassifier`
- max_iter=400, max_depth=7, learning_rate=0.08
- 파일 단위 train/val/test split (70/15/15)
- Early stopping on val

**저장**:
- `models/saved/{run_id}/model.joblib`
- `feature_pipeline.joblib`
- `config.json`, `metrics.json`
- MLOps DB에 experiment row 기록

---

## API 문서

### `GET /health`
헬스체크. 서버 생존 확인용.

### `POST /api/classify`
**요청**
```json
{
  "file_id": "string (optional)",
  "entities": [
    {
      "entity_id": "string",
      "entity_type": "LINE | LWPOLYLINE | ARC | ...",
      "raw_layer": "string",
      "length": 3.5,
      "bbox_width": 3.5,
      "bbox_height": 0.01,
      "aspect_ratio": 350
    }
  ],
  "log_predictions": true
}
```

**응답**
```json
{
  "model_version": "v_20260425_011526_2ea790",
  "total_entities": 100,
  "class_counts": {"wall": 35, "door": 5, "window": 12, "other": 48},
  "average_confidence": 0.98,
  "predictions": [
    {
      "entity_id": "1",
      "raw_layer": "WALL",
      "entity_type": "LINE",
      "predicted_class": "wall",
      "confidence": 1.0
    }
  ],
  "layer_decisions": {
    "WALL": "wall",
    "DOOR": "door",
    "WINDOW-ASAAS-0025": "window"
  }
}
```

### `GET /api/mlops/experiments`
전체 학습 실험 목록.

### `GET /api/mlops/experiments/{run_id}`
특정 실험의 상세 (하이퍼파라미터, train/val/test 메트릭, confusion matrix).

### `GET /api/mlops/models/active`
현재 활성 모델 정보.

### `POST /api/mlops/deploy`
특정 `run_id`를 active로 승격.
```json
{ "run_id": "v_20260425_011526_2ea790", "environment": "production" }
```

---

## MLOps 설계

상세 설계는 [docs/MLOPS_DESIGN.md](docs/MLOPS_DESIGN.md) 참고.

### 데이터 모델 (SQLite)

- `experiments` — 학습 run 1회 = 1 row (하이퍼파라미터, 모델 경로, 학습 시간)
- `metrics` — run별 split(train/val/test)별 지표 (accuracy, f1_macro, per-class, confusion matrix)
- `predictions` — 추론 로그 (20% 샘플링)
- `deployments` — 활성 모델 지정 + 이력

### 재학습 트리거 규칙

1. 누적 신규 DXF ≥ 학습셋의 20%
2. 최근 7일 평균 confidence가 학습 시의 90% 이하
3. 학습 시 못 본 신규 레이어명이 최근 예측의 30% 이상
4. 수동 `POST /api/mlops/retrain`

### 모델 비교

- **Golden test 기반**: 수동 라벨링된 고정 평가셋으로 모든 모델 평가 (예정)
- **Shadow inference**: active 바꾸지 않고 병렬 추론 후 결과 diff 로그 (v2)

---

## 성능 및 결과

### 데이터셋
- 크롤링된 DXF 98개 (영문 평면도 중심, 115MB)
- parse_dxf 성공률: **100%** (98/98)
- 평면도 분리 후 **101개 CSV** (다중 평면도 자동 분리)
- 총 **189,621 entities**

### 학과 vLLM Vision 평면도 검출

| 항목 | 값 |
|---|---|
| 검출 성공 | **77/80 = 96%** (토큰 한도로 18개 미호출) |
| 다중 평면도 분리 | ✓ (3개 파일에서 floorplans=2) |
| 평균 응답 시간 | 4초/파일 |
| 총 토큰 소모 | ~80k (100k 한도 내) |

### 분류 모델 성능

| Split | N | Accuracy | F1 Macro | F1 Weighted | Mean Conf |
|---|---|---|---|---|---|
| Train | 120,936 | 1.000 | 1.000 | 1.000 | 1.000 |
| Val | 24,204 | 0.999 | 0.994 | 0.999 | 1.000 |
| **Test** | **44,481** | **1.000** | **0.997** | **1.000** | 1.000 |

> Test 정확도 1.000은 "weak label 기준" 값. 실제 사람 정답 기준은 weak label 품질 한계로 70~80% 추정.

### 일반화 테스트 (학습 시 못 본 레이어명)

| 입력 | 예상 | 실제 | 평가 |
|---|---|---|---|
| `BoundaryWall_Main` | wall | **wall (1.00)** | ✓ |
| `sliding_door_new` | door | other (0.85) | ✗ |
| `Furniture_sofa` | other | **other (1.00)** | ✓ |
| 한글 `외벽선_주벽` | wall | other | ✗ |

6/8 = **75% 일반화 성능**. `BoundaryWall_Main` 같은 **학습 시 전혀 못 본 영문 표기도 wall로 분류**되는 점이 핵심 성과.

---

## 한계 및 향후 계획

### 현재 한계

1. **Weak label 의존** — 모델이 weak_label.py의 키워드 규칙을 완벽 재현. 키워드 놓치는 케이스는 모델도 놓침.
2. **다국어 취약** — 한글(`외벽`, `문`, `창`) / 스페인어(`Muro`, `puerta`) 레이어 처리 약함.
3. **Door class imbalance** — door가 전체의 1.7%로 극소수. 학습 약함.
4. **Golden test set 부재** — 절대 정확도 측정 불가.

### v2 로드맵

- [ ] **vLLM Text/Embedding 기반 re-labeling** — 다국어 + 표기 변이 흡수
- [ ] **Golden test set 구축** (수동 라벨링 10개 DXF) — 절대 정확도 측정
- [ ] **Active learning** — confidence < 0.7 예측만 수동 검수
- [ ] **Class weighting / oversampling** — door imbalance 완화
- [ ] **Shadow inference 배포 전략**
- [ ] **AWS building_cesium 연동 시연**

### v3 이후 (확장)

- **Graph Neural Network** — 엔티티 간 공간 관계 활용 (mixed layer 해결)
- **로컬 임베딩** — `sentence-transformers` + 학과 GPU로 API 의존도 제거
- **Vision 기반 CNN 앙상블** — 엔티티 주변 패치 이미지 + 지오메트리 결합

---

## 팀 정보

**Team 건치 (Geonchi)** — 2026 종합설계프로젝트
- 지도교수: 이병문 교수 (가천대학교 컴퓨터공학과)
- 팀장: 신재훈 (AI 서버 구조 설계)
- 팀원: 김상현
- 팀원: 서민혁 (개발 목표/범위, 데이터분석 요구사항)
- 팀원: **이호민** (AI 모듈 구현, 서비스 구성, API/DB 설계)

### 관련 프로젝트

- [Team_Gunchi (메인)](https://github.com/LeeHome2/Team_Gunchi) — `building_cesium` 3D 건물 배치 시스템
- [Team_Gunchi_classifier (본 리포)](https://github.com/LeeHome2/Team_Gunchi_classifier) — AI 레이어 분류 모듈

---

## 라이선스

MIT License. 자유롭게 사용 가능.

---

## 참고 자료

- [ezdxf 공식 문서](https://ezdxf.readthedocs.io/)
- [scikit-learn HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [Qwen3 Vision (학과 vLLM)](https://qwenlm.github.io/)
- [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
