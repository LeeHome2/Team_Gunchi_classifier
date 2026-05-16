"""
MLOps 레지스트리 CRUD.

experiments / metrics / deployments 에 대한 read + write 헬퍼.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlops.db import get_conn, init_db  # noqa: E402


# ─── EXPERIMENTS ─────────────────────────────────────────────
def record_experiment(
    run_id: str,
    model_type: str,
    hyperparams: Dict[str, Any],
    model_path: str,
    metrics: Dict[str, Dict[str, Any]],
    train_info: Optional[Dict[str, Any]] = None,
    features_used: Optional[Dict[str, Any]] = None,
    training_time_s: Optional[float] = None,
    status: str = "completed",
) -> None:
    """학습 완료 후 experiment + metrics 기록."""
    init_db()

    tt_s = training_time_s
    if tt_s is None and train_info:
        tt_s = train_info.get("training_time_seconds")

    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO experiments
              (run_id, model_type, hyperparams, features_used, model_path,
               training_time_s, status, train_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                model_type,
                json.dumps(hyperparams, ensure_ascii=False),
                json.dumps(features_used or {}, ensure_ascii=False),
                model_path,
                tt_s,
                status,
                json.dumps(train_info or {}, ensure_ascii=False, default=str),
            ),
        )

        # split별 metric
        for split_name, m in metrics.items():
            conn.execute(
                """
                INSERT INTO metrics
                  (run_id, split, accuracy, f1_macro, f1_weighted,
                   per_class, confusion_matrix, confidence_mean, n_samples)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    split_name,
                    m.get("accuracy"),
                    m.get("f1_macro"),
                    m.get("f1_weighted"),
                    json.dumps(m.get("per_class", {}), ensure_ascii=False),
                    json.dumps(m.get("confusion_matrix", []), ensure_ascii=False),
                    m.get("confidence_mean"),
                    m.get("n"),
                ),
            )


def list_experiments(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT run_id, model_type, status, created_at, training_time_s, model_path
            FROM experiments
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_experiment(run_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        exp = conn.execute(
            "SELECT * FROM experiments WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not exp:
            return None
        result = dict(exp)
        # JSON 필드 파싱
        for k in ("hyperparams", "features_used", "train_info"):
            if result.get(k):
                try:
                    result[k] = json.loads(result[k])
                except Exception:
                    pass

        # 메트릭 조회
        metrics_cur = conn.execute(
            "SELECT * FROM metrics WHERE run_id = ? ORDER BY id", (run_id,)
        )
        metrics = {}
        for row in metrics_cur.fetchall():
            d = dict(row)
            for k in ("per_class", "confusion_matrix"):
                if d.get(k):
                    try:
                        d[k] = json.loads(d[k])
                    except Exception:
                        pass
            metrics[d["split"]] = d
        result["metrics"] = metrics
        return result


def register_experiment(
    run_id: str,
    model_type: str,
    status: str = "completed",
    train_info: Optional[Dict[str, Any]] = None,
) -> None:
    """간단한 experiment 등록 (업로드된 모델 등 metrics 없는 경우)."""
    init_db()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO experiments
              (run_id, model_type, hyperparams, features_used, model_path,
               training_time_s, status, train_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                model_type,
                json.dumps({}, ensure_ascii=False),
                json.dumps({}, ensure_ascii=False),
                "",  # model_path는 비워둠 (기본 경로 사용)
                None,
                status,
                json.dumps(train_info or {}, ensure_ascii=False, default=str),
            ),
        )


# ─���─ DEPLOYMENTS ──���──────────────────────────────────────────
def set_active(run_id: str, environment: str = "production", notes: str = "") -> None:
    """지정 run_id를 active로 승격. 같은 environment의 이전 active는 비활성화."""
    init_db()
    with get_conn() as conn:
        conn.execute(
            "UPDATE deployments SET is_active = 0 WHERE environment = ? AND is_active = 1",
            (environment,),
        )
        conn.execute(
            """
            INSERT INTO deployments (run_id, environment, is_active, notes)
            VALUES (?, ?, 1, ?)
            """,
            (run_id, environment, notes),
        )


def get_active(environment: str = "production") -> Optional[Dict[str, Any]]:
    """현재 active 모델의 run_id + model_path."""
    init_db()
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT d.run_id, d.environment, d.deployed_at, e.model_path, e.model_type
            FROM deployments d
            JOIN experiments e ON e.run_id = d.run_id
            WHERE d.environment = ? AND d.is_active = 1
            ORDER BY d.id DESC
            LIMIT 1
            """,
            (environment,),
        ).fetchone()
        return dict(row) if row else None


# ─── PREDICTIONS (샘플링 로그) ────────────────────────────────
def log_predictions(
    run_id: str,
    predictions: List[Dict[str, Any]],
) -> None:
    """추론 결과 일부를 샘플링해서 기록."""
    init_db()
    if not predictions:
        return
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO predictions
              (run_id, file_id, entity_id, raw_layer, entity_type,
               predicted_class, confidence, is_new_layer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    p.get("file_id"),
                    p.get("entity_id"),
                    p.get("raw_layer"),
                    p.get("entity_type"),
                    p.get("predicted_class"),
                    p.get("confidence"),
                    int(p.get("is_new_layer", False)),
                )
                for p in predictions
            ],
        )
