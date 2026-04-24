# Quickstart — 로컬에서 학습 + 테스트

## 파이프라인 전체 흐름

```
DXF 98개
  ├─[1] parse_dxf       → data/processed/*.csv + *.meta.json
  ├─[2] render_preview  → data/preview/*.png + *.preview.json
  ├─[3] detect_floorplan → 학과 vLLM Vision → data/processed/*.bboxes.json
  ├─[4] crop_entities   → bbox로 엔티티 필터링 → data/cropped/*.csv
  └─[5] weak_label      → 키워드/지오메트리 라벨링 → data/labeled/*.csv
                                                       ↓
[6] train → HistGradientBoosting → models/saved/{run_id}/
                                                       ↓
[7] set_active → mlops.db에 active 지정
                                                       ↓
[8] main.py (FastAPI) → /api/classify 서빙
```

## 0. 사전 준비 (한 번만)

```powershell
cd C:\Users\user\Desktop\26-1\ai_layer_classifier

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 1. 학과 vLLM 연결 확인

학과 vLLM이 학외 접속 가능한지 먼저 확인:

```powershell
python -m tests.smoke_test_llm
```

모든 테스트 (text/embedding/vision) PASS 확인. 실패 시:
- VPN 또는 SSH 포트 포워딩 필요할 수 있음
- 또는 학내망 PC에서 실행

## 2. 전체 학습 데이터 빌드 (orchestrator)

```powershell
# 실제 vLLM 호출 (20~30분 소요, 98개 파일 × Vision API)
python -m dataset.build_training_dataset --dxf-dir "..\데이터셋1-dxf\dxf"

# 또는 vLLM 없이 mock 검증 (빠름, 평면도 분리 없이 전체 DXF 사용)
python -m dataset.build_training_dataset --dxf-dir "..\데이터셋1-dxf\dxf" --mock

# 5개만 테스트
python -m dataset.build_training_dataset --dxf-dir "..\데이터셋1-dxf\dxf" --mock --limit 5
```

각 단계는 idempotent — 중간에 멈춰도 기존 결과 재활용해서 이어서 실행 가능.

### 각 단계 개별 실행 (고급)

orchestrator 없이 단계별로:

```powershell
python -m dataset.parse_dxf --input-dir "..\데이터셋1-dxf\dxf"
python -m dataset.render_preview --input-dir "..\데이터셋1-dxf\dxf"
python -m dataset.detect_floorplan --input-dir data\preview
python -m dataset.crop_entities_by_bbox --processed-dir data\processed --preview-dir data\preview --output-dir data\cropped
python -m dataset.weak_label --input-dir data\cropped --output-dir data\labeled
```

## 3. 학습

```powershell
python -m training.train --input-dir data\labeled
```

출력:
- `models/saved/{run_id}/model.joblib`
- `models/saved/{run_id}/feature_pipeline.joblib`
- `models/saved/{run_id}/config.json`
- `models/saved/{run_id}/metrics.json`
- `mlops.db`에 experiment + metrics 기록

## 4. Active 모델 지정

```python
python -c "from mlops.registry import set_active; set_active('<run_id>')"
```

또는 API 서버 띄우고 `POST /api/mlops/deploy`.

## 5. FastAPI 서버 실행

```powershell
python main.py
# 또는
uvicorn main:app --host 0.0.0.0 --port 8001
```

`http://localhost:8001/docs` Swagger UI.

## 6. 동작 확인

```powershell
curl http://localhost:8001/health
curl http://localhost:8001/api/mlops/experiments
curl http://localhost:8001/api/mlops/models/active

curl -X POST http://localhost:8001/api/classify `
  -H "Content-Type: application/json" `
  -d '{"file_id":"test","entities":[{"entity_id":"1","entity_type":"LINE","raw_layer":"WALL","length":3.5,"bbox_width":3.5,"bbox_height":0.01,"aspect_ratio":350}]}'
```

## 샌드박스 검증 결과 (2026-04-23)

### Mock 모드 전체 파이프라인 (5개 파일)
```
[1/5] parse_dxf        → OK=5 FAIL=0
[2/5] render_preview   → OK=5 FAIL=0
[3/5] detect(mock)     → OK=5 (0.0s)
[4/5] crop             → OK=5, 7,285 entities
[5/5] weak_label       → OK=5
       분포: wall=475 door=136 window=167 other=6507
```

### 학습 (20개 파일 기반, mock 모드 없이 직접 실행했을 때)
| split | accuracy | f1_macro | f1_weighted |
|---|---|---|---|
| train | 1.000 | 1.000 | 1.000 |
| val | 1.000 | 0.999 | 1.000 |
| **test** | **0.945** | **0.896** | **0.955** |

### 추론 검증 (학과 vLLM 없이 TF-IDF만)
- `WALL`, `DOORS`, `WINDOW-ASAAS-0025` → conf 1.0 정확
- `ExteriorWall_Ground` (학습 시 못 본 영문) → wall conf 0.955 (일반화 성공)
- `Muro` (스페인어) → other로 오분류 (학습 샘플 부족 / embedding 없음)

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: sklearn/joblib/fastapi/...` | 미설치 | `pip install -r requirements.txt` |
| `disk I/O error` (SQLite) | 마운트 이슈 | `set MLOPS_DB_PATH=C:\temp\mlops.db` |
| `활성 모델이 없습니다` | set_active 안 함 | step 4 실행 |
| `Connection refused` (vLLM) | 네트워크 차단 | VPN or SSH 포트포워딩 or `--mock` |
| `401 Unauthorized` (vLLM) | API 키 오타 | `config.py` 확인 |
| Muro/다국어 오분류 | TF-IDF 한계 | 월요일 이후 embedding 교체 |

## 학과 서버 배포 (월요일 이후)

```bash
# 로컬에서 학습 완료 후
scp -P <port> -r ai_layer_classifier/ t26206@ceprj2.gachon.ac.kr:~/
# 또는 VSCode Remote-SSH 드래그앤드롭

# 학과 서버에서
cd ~/ai_layer_classifier
# 조교 요청한 라이브러리 설치 완료 후
pip install --user -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port <할당포트>
```

## 전체 파일 목록

```
ai_layer_classifier/
├── config.py                           # 경로/LLM/분류/MLOps 설정
├── main.py                             # FastAPI 서빙
├── requirements.txt
├── QUICKSTART.md (이 문서)
├── README.md
│
├── dataset/
│   ├── parse_dxf.py                    # [1] DXF → CSV
│   ├── render_preview.py               # [2] DXF → PNG
│   ├── detect_floorplan.py             # [3] 학과 vLLM Vision → bbox
│   ├── crop_entities_by_bbox.py        # [4] bbox로 엔티티 필터
│   ├── weak_label.py                   # [5] 키워드 라벨링
│   └── build_training_dataset.py       # 1~5 orchestrator
│
├── training/
│   ├── feature_extractor.py            # TF-IDF + one-hot + 지오메트리
│   └── train.py                        # [6] HistGradientBoosting
│
├── inference/
│   └── predictor.py                    # 모델 로드 + 예측
│
├── mlops/
│   ├── db.py                           # SQLite 스키마
│   └── registry.py                     # experiments/deployments CRUD
│
├── llm/
│   ├── client.py                       # 학과 vLLM 래퍼
│   └── parse_response.py               # JSON 응답 검증
│
├── tests/
│   ├── smoke_test_llm.py               # vLLM 3모델 연결 테스트
│   ├── test_layer_labeling_llm.py      # LLM 라벨링 실험
│   └── README.md                       # 테스트 가이드
│
├── configs/
│   ├── dataset_meta.json
│   ├── train_params.json
│   └── llm_prompts/floorplan_bbox_prompt.txt
│
├── data/
│   ├── processed/      # [1,3] CSV + bboxes JSON
│   ├── preview/        # [2] PNG
│   ├── cropped/        # [4] 평면도별 CSV
│   ├── labeled/        # [5] weak_label 추가
│   └── reports/        # 빌드 리포트, 분석 리포트
│
├── models/saved/       # [6] 학습된 모델들
├── mlops.db            # SQLite
│
└── docs/
    ├── MLOPS_DESIGN.md
    └── (기타 설계 문서)
```
