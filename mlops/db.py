"""
SQLite 연결 + 스키마 초기화.

경량 사용: SQLAlchemy 없이 sqlite3 직접 사용 (학과 서버에 이미 내장).
WAL 모드로 reader/writer 병행 가능.
"""
from __future__ import annotations

import sqlite3
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MLOPS_DB_PATH  # noqa: E402


# 스레드 안전성: FastAPI는 비동기라 sqlite3 연결을 스레드별로
_local = threading.local()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
  run_id            TEXT PRIMARY KEY,
  model_type        TEXT NOT NULL,
  hyperparams       TEXT NOT NULL,          -- JSON
  features_used     TEXT,                   -- JSON
  model_path        TEXT NOT NULL,          -- 상대/절대 경로
  training_time_s   REAL,
  status            TEXT NOT NULL DEFAULT 'completed',
  error_message     TEXT,
  train_info        TEXT,                   -- JSON (파일 분할 정보 등)
  created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS metrics (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id            TEXT NOT NULL,
  split             TEXT NOT NULL,          -- 'train' | 'val' | 'test' | 'golden'
  accuracy          REAL,
  f1_macro          REAL,
  f1_weighted       REAL,
  per_class         TEXT,                   -- JSON
  confusion_matrix  TEXT,                   -- JSON
  confidence_mean   REAL,
  n_samples         INTEGER,
  created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (run_id) REFERENCES experiments(run_id)
);

CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);

CREATE TABLE IF NOT EXISTS predictions (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id            TEXT,
  file_id           TEXT,
  entity_id         TEXT,
  raw_layer         TEXT,
  entity_type       TEXT,
  predicted_class   TEXT,
  confidence        REAL,
  is_new_layer      INTEGER DEFAULT 0,
  created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);

CREATE TABLE IF NOT EXISTS deployments (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id            TEXT NOT NULL,
  environment       TEXT DEFAULT 'production',
  is_active         INTEGER DEFAULT 1,
  notes             TEXT,
  deployed_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (run_id) REFERENCES experiments(run_id)
);

CREATE INDEX IF NOT EXISTS idx_deployments_active ON deployments(is_active, environment);
"""


def _connect() -> sqlite3.Connection:
    """스레드 로컬 연결 반환. WAL 모드 + foreign_keys 활성."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn

    MLOPS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(MLOPS_DB_PATH),
        check_same_thread=False,
        isolation_level=None,  # autocommit
        timeout=30.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    _local.conn = conn
    return conn


def init_db() -> None:
    """스키마 생성 (idempotent)."""
    conn = _connect()
    conn.executescript(SCHEMA_SQL)


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """with 문에서 쓸 수 있는 연결 컨텍스트."""
    conn = _connect()
    try:
        yield conn
    except Exception:
        raise


if __name__ == "__main__":
    init_db()
    print(f"MLOps DB 초기화: {MLOPS_DB_PATH}")
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        print("테이블 목록:")
        for row in cur.fetchall():
            print(f"  - {row['name']}")
