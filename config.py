"""
ai_layer_classifier 전역 설정
- 디렉토리 경로
- 학과 vLLM 서버 설정
- 분류 클래스 정의

.env 파일이 프로젝트 루트에 있으면 자동 로드됨.
커밋 금지 (.gitignore에 등록됨).
"""
import os
import warnings
from pathlib import Path

# ─── 프로젝트 루트 ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ─── .env 자동 로드 (있을 때만) ────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = BASE_DIR / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv 미설치 시 환경변수 직접 export로 대응

# ─── 데이터 폴더 ───────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # 원본 DXF 파일 (dataset1-dxf 심볼릭 링크)
PREVIEW_DIR = DATA_DIR / "preview"      # DXF → PNG 렌더 결과 + extents JSON
PROCESSED_DIR = DATA_DIR / "processed"  # 파싱된 CSV, bbox 결과 캐시
SPLIT_DIR = DATA_DIR / "splits"         # train/val/test split
REPORT_DIR = DATA_DIR / "reports"       # 데이터셋 리포트

# ─── 설정 / 모델 ───────────────────────────────────────────
CONFIG_DIR = BASE_DIR / "configs"
PROMPT_DIR = CONFIG_DIR / "llm_prompts"
MODEL_DIR = BASE_DIR / "models" / "saved"

DATASET_META_PATH = CONFIG_DIR / "dataset_meta.json"
TRAIN_PARAMS_PATH = CONFIG_DIR / "train_params.json"
FLOORPLAN_PROMPT_PATH = PROMPT_DIR / "floorplan_bbox_prompt.txt"

# ─── 학과 vLLM 프록시 (Gachon cellm) ───────────────────────
# 모델 별칭: "text" | "vision" | "embedding"
# 제약: 1회 요청 32,000 토큰 상한, 이미지 2~5MB 권장
#
# API 키는 환경변수 또는 .env 파일로 반드시 설정. 하드코딩 금지.
# .env.example 복사 후 .env에 본인 키 입력.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://cellm.gachon.ac.kr:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TEXT_MODEL = os.getenv("LLM_TEXT_MODEL", "text")
LLM_VISION_MODEL = os.getenv("LLM_VISION_MODEL", "vision")
LLM_EMBEDDING_MODEL = os.getenv("LLM_EMBEDDING_MODEL", "embedding")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

if not LLM_API_KEY:
    warnings.warn(
        "LLM_API_KEY 환경변수가 설정되지 않았습니다. "
        "vLLM 호출(detect_floorplan 등) 시 실패합니다. "
        ".env 파일 또는 export LLM_API_KEY=... 로 설정하세요.",
        RuntimeWarning,
    )

# ─── 분류 클래스 (4-class) ─────────────────────────────────
# 외벽+내벽 합쳐서 wall 하나로. door/window/other.
CLASSES = ["wall", "door", "window", "other"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}

# ─── 엔티티 타입 화이트리스트 ──────────────────────────────
# 벽/문/창으로 분류될 수 있는 엔티티 타입.
# TEXT / MTEXT / DIMENSION / HATCH / INSERT 등은 자동으로 other.
GEOMETRIC_ENTITY_TYPES = {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE"}

# ─── 외부 경로 ─────────────────────────────────────────────
# 팀원이 크롤링한 원본 데이터셋 (98개 DXF)
EXTERNAL_DATASET_DIR = Path(os.getenv(
    "EXTERNAL_DATASET_DIR",
    str(BASE_DIR.parent / "데이터셋1-dxf" / "dxf"),
))

# ─── MLOps / 서빙 설정 ─────────────────────────────────────
MLOPS_DB_PATH = Path(os.getenv("MLOPS_DB_PATH", str(BASE_DIR / "mlops.db")))
SERVING_HOST = os.getenv("SERVING_HOST", "0.0.0.0")
SERVING_PORT = int(os.getenv("SERVING_PORT", "8001"))

# 추론 시 GPU 사용 여부 (기본: CPU. 학과 서버 규정 준수)
INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "cpu")

# ─── 폴더 자동 생성 ────────────────────────────────────────
for path in [
    DATA_DIR, RAW_DIR, PREVIEW_DIR, PROCESSED_DIR, SPLIT_DIR, REPORT_DIR,
    CONFIG_DIR, PROMPT_DIR, MODEL_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
