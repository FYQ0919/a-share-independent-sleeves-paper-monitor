import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from app.models import RunResult


class Storage:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    market_status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    report_markdown TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_states (
                    strategy_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_decisions (
                    strategy_id TEXT NOT NULL,
                    signal_date TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(strategy_id, signal_date)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS backtests (
                    backtest_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    config TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    def save_run(self, result: RunResult):
        payload = result.to_dict()
        report_markdown = payload.pop("report_markdown")
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO runs(run_id, created_at, mode, market_status, payload, report_markdown) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    result.run_id,
                    result.created_at.isoformat(),
                    result.mode,
                    result.market_status,
                    json.dumps(payload, ensure_ascii=False),
                    report_markdown,
                ),
            )

    def latest(self) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
        return self._row(row) if row else None

    def history(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(row) for row in rows]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._row(row) if row else None

    def save_backtest(self, result: Dict[str, Any]):
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO backtests(backtest_id, created_at, config, payload) VALUES (?, ?, ?, ?)",
                (
                    result["backtest_id"],
                    result["created_at"],
                    json.dumps(result["config"], ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                ),
            )

    def latest_backtest(self) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM backtests ORDER BY created_at DESC LIMIT 1").fetchone()
        return json.loads(row["payload"]) if row else None

    def backtest_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM backtests ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def load_strategy_state(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM strategy_states WHERE strategy_id = ?", (strategy_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def strategy_decision_history(self, strategy_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM strategy_decisions WHERE strategy_id = ? ORDER BY signal_date DESC LIMIT ?",
                (strategy_id, limit),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def save_strategy_snapshot(self, strategy_id: str, signal_date: str, state: Dict, decision: Dict):
        state_payload = json.dumps(state, ensure_ascii=False)
        decision_payload = json.dumps(decision, ensure_ascii=False)
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO strategy_states(strategy_id, updated_at, payload) VALUES (?, ?, ?)",
                (strategy_id, signal_date, state_payload),
            )
            connection.execute(
                "INSERT OR REPLACE INTO strategy_decisions(strategy_id, signal_date, payload) VALUES (?, ?, ?)",
                (strategy_id, signal_date, decision_payload),
            )

    @staticmethod
    def _row(row: sqlite3.Row) -> Dict[str, Any]:
        payload = json.loads(row["payload"])
        payload["report_markdown"] = row["report_markdown"]
        return payload
