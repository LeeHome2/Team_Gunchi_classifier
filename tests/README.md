# 동작 테스트 가이드

## 학과 서버로 코드 옮기기 (최초 1회)

```bash
# 로컬에서 압축
cd C:\Users\user\Desktop\26-1
tar -czf ai_layer_classifier.tgz ai_layer_classifier/ \
    --exclude='**/__pycache__' --exclude='**/data/preview/*' \
    --exclude='**/data/processed/*' --exclude='**/data/splits/*'

# 서버로 전송 (새 터미널에서)
scp -P 65006 ai_layer_classifier.tgz t26206@ceprj2.gachon.ac.kr:~/

# 서버에 SSH 접속 후
ssh t26206@ceprj2.gachon.ac.kr -p 65006
# 비밀번호: Thisisgunch@@@
cd ~
tar -xzf ai_layer_classifier.tgz
cd ai_layer_classifier
```

**대안**: VSCode Remote-SSH 쓰고 있으면 폴더 드래그앤드롭으로 업로드.

## 패키지 설치 (서버에서)

학과 서버엔 이미 대부분 깔려 있음. 추가 설치만:

```bash
pip install --user ezdxf xgboost pillow joblib python-multipart
# matplotlib, pandas, numpy, openai, fastapi, uvicorn, sqlalchemy, sklearn, torch는 시스템에 있음
```

## 단계별 테스트 (순서대로)

### Step 1: 학과 vLLM 연결 스모크 테스트 (1분)

```bash
cd ~/ai_layer_classifier
python3 -m tests.smoke_test_llm
```

**기대 출력**:
```
[1/3] TEXT 모델
  응답 (1.xxs): 'OK'
  PASS
[2/3] EMBEDDING 모델
  응답 (0.xxs): 차원=768, ...
  PASS
[3/3] VISION 모델
  응답 (2.xxs): '{"has_floorplan": true}'
  PASS
```

**실패 시 체크**:
- 401 → API 키 잘못됨 (`config.py` 확인)
- 404 → base_url 잘못됨
- Timeout → 서버 장애 또는 네트워크
- Model not found → 별칭 `text/vision/embedding` 대신 다른 이름 확인 필요

### Step 2: LLM 레이어 분류 능력 검증 (2분)

**이게 핵심**. LLM이 레이어명만 보고 wall/door/window/other를 얼마나 잘 맞추는지 측정. 수동 라벨링 범위가 이 결과에 따라 결정됨.

```bash
python3 -m tests.test_layer_labeling_llm
```

**기대 결과 해석**:
- 정확도 ≥ 90% → LLM으로 weak labeling 확정. 수동은 Golden test 10개만
- 정확도 80~90% → LLM + 낮은 confidence만 수동 검수
- 정확도 65~80% → LLM + 키워드 앙상블, 수동 많이 필요
- 정확도 < 65% → 기존 키워드 매칭 + 전면 수동

결과는 `data/reports/llm_labeling_experiment.json`에 저장됨.

### Step 3: DXF → PNG 렌더링 (1분, 1개 샘플)

```bash
python3 -m dataset.render_preview \
    -i ~/데이터셋1-dxf/dxf/2bhk_house_design_ground_floor_plan_op_1.dxf \
    --dpi 150
```

**확인 사항**:
- `data/preview/*.png` 생성됨
- 벽체가 검은색으로 **잘 보임** (흰 배경에 흰선 문제 없는지)
- 파일 크기 5MB 이내 (vision LLM 제약)
- `data/preview/*.preview.json`에 `extents_cad` 저장됨

VSCode Remote로 이 PNG 열어서 실제 평면도 모양이 보이는지 눈으로 확인.

### Step 4: Vision LLM으로 평면도 bbox 검출 (수동 테스트)

Step 3에서 만든 PNG 1개에 대해 Python 인터프리터에서:

```python
from llm.client import ask_vision
from llm.parse_response import extract_json_from_text, validate_floorplan_response
from pathlib import Path

# 프롬프트 로드
prompt = Path("configs/llm_prompts/floorplan_bbox_prompt.txt").read_text(encoding="utf-8")

# 호출
response_str = ask_vision(
    image_path="data/preview/2bhk_house_design_ground_floor_plan_op_1.png",
    prompt=prompt,
    system="너는 건축 CAD 도면에서 평면도 영역을 찾는 도우미이며, 반드시 JSON으로만 응답해야 한다.",
)
print("RAW 응답:", response_str)

# 파싱
parsed = extract_json_from_text(response_str)
validated = validate_floorplan_response(parsed)
print("검증됨:", validated)
```

**기대**: `floorplans_found: true`, bbox 좌표가 평면도 영역과 일치

### Step 5 (선택): parse_dxf 배치 테스트

```bash
# 98개 전체 파싱
python3 -m dataset.parse_dxf --input-dir ~/데이터셋1-dxf/dxf

# 또는 처음 5개만
python3 -m dataset.parse_dxf --input-dir ~/데이터셋1-dxf/dxf --limit 5
```

출력: `data/processed/*.csv` + `*.meta.json`

## 결과 회수 (서버 → 로컬)

JSON 리포트들은 가볍게 scp로:

```bash
# 로컬에서
scp -P 65006 -r t26206@ceprj2.gachon.ac.kr:~/ai_layer_classifier/data/reports/ ./server_results/
```

## 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: openai` | `pip install --user` 안 함 | requirements 재설치 |
| `Connection refused` | 학과 네트워크 일시 장애 | 잠시 후 재시도 |
| `401 Unauthorized` | API 키 오타 | `config.py` 또는 환경변수 확인 |
| `model not found` | 모델 별칭 변경됨 | `/v1/models` 쿼리해서 실제 이름 확인 |
| PNG에 벽이 안 보임 | 흰 배경에 흰선 | 최신 render_preview (config 포함) 사용 확인 |
| PNG 크기 > 5MB | dpi 너무 높음 | `--dpi 100`으로 낮추기 |
