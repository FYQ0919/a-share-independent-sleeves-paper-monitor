from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
from threading import Lock

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.backtest_data import BaoStockHistoryProvider
from app.config import settings
from app.factor_mining import FactorMiningConfig, MiningWindow
from app.storage import Storage
from run_factor_mining import run


_UPDATE_LOCK = Lock()


def rolling_factor_config(history: pd.DataFrame) -> FactorMiningConfig:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(history["date"]).dropna().unique()))
    required = 756 + 504 + 252
    if len(dates) < required:
        raise RuntimeError(f"滚动因子更新至少需要{required}个交易日，当前只有{len(dates)}个")
    diagnostic = dates[-252:]
    selection = dates[-(252 + 504):-252]
    discovery = dates[-required:-(252 + 504)]
    return FactorMiningConfig(
        discovery=MiningWindow(
            "rolling_discovery",
            discovery[0].date(),
            discovery[-1].date(),
            "rolling factor discovery",
        ),
        selection=MiningWindow(
            "rolling_selection",
            selection[0].date(),
            selection[-1].date(),
            "independent rolling selection",
        ),
        diagnostic=MiningWindow(
            "rolling_quarantine",
            diagnostic[0].date(),
            diagnostic[-1].date(),
            "latest quarantine; deployment blocker only",
        ),
    )


def _latest_candidates(storage: Storage) -> tuple[list[dict], str]:
    eligible = [row for row in storage.history(200) if len(row.get("candidates", [])) >= 50]
    if not eligible:
        raise RuntimeError("没有可用于自动因子更新的Top50快照")
    latest = max(eligible, key=lambda row: pd.Timestamp(row["created_at"]))
    return latest["candidates"][:50], latest["run_id"]


def run_auto_update(
    as_of: date | None = None,
    refresh_history: bool = True,
) -> dict:
    if not _UPDATE_LOCK.acquire(blocking=False):
        raise RuntimeError("已有自动因子更新任务正在运行")
    try:
        storage = Storage(settings.database_path)
        candidates, snapshot_run_id = _latest_candidates(storage)
        codes = [row["code"] for row in candidates]
        names = {row["code"]: row["name"] for row in candidates}
        target_end = as_of or date.today()
        provider = BaoStockHistoryProvider(settings.cache_dir)
        if refresh_history:
            history, warnings = provider.load(codes, names, date(2016, 8, 25), target_end)
        else:
            from optimize_factor_strategy import load_current_history

            history, _, warnings = load_current_history(candidates)
        history["date"] = pd.to_datetime(history["date"])
        history = history[history["date"].le(pd.Timestamp(target_end))].copy()
        config = rolling_factor_config(history)
        return run(
            config=config,
            history=history,
            requested_codes=codes,
            snapshot_run_id=snapshot_run_id,
            input_warnings=warnings,
            update_mode="rolling_monthly_auto",
        )
    finally:
        _UPDATE_LOCK.release()


def main() -> None:
    payload = run_auto_update()
    print({
        "status": payload["status"],
        "registry_version": payload["registry_version"],
        "generated_factor_count": payload["generated_factor_count"],
        "selected_generated_factor_count": payload["selected_generated_factor_count"],
        "production_change": payload["production_change"],
    })


if __name__ == "__main__":
    main()
