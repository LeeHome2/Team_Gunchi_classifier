# 데이터셋1 구성 분석 리포트

> 분석일: 2026-04-23
> 대상: `C:\Users\user\Desktop\26-1\데이터셋1-dxf\dxf\` (팀원 크롤링 데이터 98개)
> 도구: ezdxf 1.4.3

## 1. 파일 단위 요약

| 항목 | 값 |
|---|---|
| 총 파일 수 | 98 |
| 파싱 성공률 | 100% (98/98) |
| DXF 버전 | AC1027 (AutoCAD 2013) 균일 |
| 총 엔티티 수 | 253,103 |
| 파일당 엔티티 수 | min=2, 중위값=1,171, max=15,243, 평균=2,583 |
| 파일당 레이어 수 | min=1, 중위값=14, max=54, 평균=17 |
| 유니크 레이어명 수 | **789** (전 파일 통합) |

## 2. 엔티티 타입 분포 (전체)

| 타입 | 개수 | 비율 | 학습 대상? |
|---|---:|---:|---|
| LINE | 127,434 | 50.3% | ✓ wall/door/window 후보 |
| LWPOLYLINE | 76,601 | 30.3% | ✓ |
| ARC | 17,881 | 7.1% | ✓ (door swing 등) |
| INSERT | 7,003 | 2.8% | ✗ 블록 참조 — 제외 |
| TEXT | 5,473 | 2.2% | ✗ other |
| MTEXT | 4,777 | 1.9% | ✗ other |
| ELLIPSE | 3,790 | 1.5% | ✓ |
| DIMENSION | 2,637 | 1.0% | ✗ other |
| CIRCLE | 2,576 | 1.0% | ✓ (기둥, fixture) |
| HATCH | 1,861 | 0.7% | ✗ 채움 패턴 — 제외 |
| 3DFACE | 1,213 | 0.5% | ✗ |
| SOLID | 1,145 | 0.5% | ✗ |
| SPLINE | 246 | 0.1% | ✓ |
| LEADER | 127 | 0.1% | ✗ |
| POINT | 119 | 0.0% | ✗ |

**시사점**: LINE + LWPOLYLINE + ARC 3종이 87.7%. 기하 feature는 이 셋에 최적화하면 됨.

## 3. 레이어명 카테고리 분류 (키워드 기반)

| 카테고리 | 엔티티 수 | 비율 |
|---|---:|---:|
| **OTHER (미분류)** | 95,102 | 37.6% |
| wall | 36,669 | 14.5% |
| hatch | 35,181 | 13.9% |
| furn | 31,440 | 12.4% |
| pdf_import | 21,195 | 8.4% |
| dim | 11,325 | 4.5% |
| text | 7,292 | 2.9% |
| window | 6,869 | 2.7% |
| stair | 3,254 | 1.3% |
| default (레이어 "0", "Defpoints") | 2,485 | 1.0% |
| door | 2,410 | 1.0% |

## 4. 핵심 관찰

### 4.1 레이어명 표기 파편화가 심함

같은 "벽"을 뜻하는 레이어가 **85가지 유니크 이름**으로 등장. 예시:
- 영문 대소문자/하이픈: `WALL`, `Wall`, `wall`, `A-WALL`, `A-Wall`, `A-Walls`, `EXT WALLS`, `I-WALL`, `INTERNALWALL`, `L WALL`, `3BWALL`
- 다국어: `Muro` (스페인어)
- 복합어: `0P-BrickWall`, `Boundary Wall`, `Compound Wall-ASAAS-0025`, `A-External Walls-G`
- 노이즈 포함: `A-waldrobe line pen 1.` (오탈자? waldrobe), `BA-A--01$0$xBDAFP01$0$a-wall-full-extr` (xref 잔해)

**→ embedding 모델의 필요성이 명확히 확인됨**. TF-IDF로는 이 표기 변이를 흡수하지 못함.

### 4.2 키워드 매칭이 놓치는 케이스

OTHER(37.6%)에 섞여있는 실제 벽/문/창 레이어:
- `WIN` (1,649), `Win` (490), `WINOWS` (1,286) — window 오타/축약
- `DOORS` (1,301), `D` (491) — door 복수/축약
- `Steps` (844), `S-STRS` (872) — stair
- `Coloumns` (612) — 기둥 오탈자

→ weak label 키워드 사전에 이런 변형들 추가 필요.

### 4.3 Mixed 레이어 — 벽과 기타가 같은 레이어에 섞임

**이게 제일 까다로움**:
- `Interior` (17,847 entities) — 실내 요소 전부. 벽+가구+문 혼재 가능성
- `Arch-plan` (13,530) — "건축도 전체"
- `A_All` (2,736), `FIXER` (2,334), `Layer1`, `Layer2` — 이름으로 판별 불가

→ 이 레이어들은 **지오메트리 기반 분류**에 전적으로 의존해야 함. 임베딩도 "Interior"만으로는 분류가 안 되니까 모델이 지오메트리 feature로 구분해야 함.

### 4.4 노이즈 레이어 — 반드시 필터링

| 레이어 패턴 | 개수 | 정체 |
|---|---:|---|
| `PDF*_*` 계열 | 21,195 | CAD에 PDF 임포트한 잔해 |
| `www.cadblocksfree.com` | 740 | 워터마크 |
| `Defpoints` | (일부) | AutoCAD 내부 표시용 |
| `0` | 2,431 | AutoCAD 기본 레이어, 실제로 INSERT 28% + 잡동사니 |

→ 전처리 단계에서 **정규식으로 제외** 필요 (예: `^PDF\d*_`, `cadblocks`, `^Defpoints$`).

### 4.5 다중 도면 의심 파일

단순 2D 히스토그램으로는 정확한 수를 세기 어려움(블록 INSERT 때문에 과다추정). 하지만 **파일명 기반**으로 다층 구성 확실한 것들:
- `3_bhk_house_design_first_floor_plan.dxf` + `3_bhk_house_design_ground_floor_plan.dxf` — 쌍으로 구성
- `...floor_plan_op_1/2/3.dxf` — 옵션 여러 개
- `steel_building_with_h_beam_metal_truss_5_.dxf` — 기계부품 상세 (평면도 아닐 수 있음)

→ 실제 다중 평면도 여부는 **render_preview.py 구현 후 vision 모델로 확인**이 현실적. 지금 섣부른 추정 X.

## 5. 전처리 규칙 초안 (weak label 엔진용)

```python
# 1단계: 노이즈 레이어 하드 필터 (무조건 other)
NOISE_PATTERNS = [
    r"^PDF\d*_",          # PDF 임포트 잔해
    r"cadblocks",         # 워터마크
    r"^Defpoints$",       # AutoCAD 내부
    r"^Layer\d+$",        # 이름 없음
]

# 2단계: 엔티티 타입 화이트리스트 (geometric only)
GEOMETRIC_TYPES = {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE"}

# 3단계: 키워드 매칭 (확장판)
KEYWORDS = {
    "wall":   ["wall", "muro", "mur", "壁", "벽", "walls"],
    "door":   ["door", "doors", "puerta", "porte", "문", "\\bd\\b"],
    "window": ["window", "windows", "ventana", "fen", "창", "wind", "win", "winow"],
}

# 4단계: 지오메트리 sanity check
# - 총 길이 < 0.5m (단위 추정 후) → other
# - 레이어당 엔티티 <= 3 → other (블록 잔해)
# - 엔티티 bbox가 도면 extents의 80% 이상 → other (외곽 테두리)
```

## 6. 다음 단계 영향

이 분석 결과를 반영해서:

1. `dataset/parse_dxf.py`는 **모든 엔티티 추출** + 엔티티별 지오메트리 메트릭 계산
2. `dataset/build_dataset.py`의 weak labeler는 **4단계 파이프라인** (위 규칙)
3. 학습 feature에 `raw_layer` 임베딩 **필수** (레이어명 파편화 때문)
4. Golden test set 구성 시 mixed 레이어(`Interior`, `Arch-plan`) 포함된 파일 일부러 섞어서 어려운 케이스 평가
