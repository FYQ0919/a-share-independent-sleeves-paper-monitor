from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.backtest_engine import BacktestEngine
from app.config import BASE_DIR
from compare_validation_nasdaq import frozen_research_candidates
from research_lgbm_multi_objective_v1 import multihead_portfolio_backtest
from research_lgbm_new_factors_v1 import build_extended_feature_frame
from research_qlib_lgbm_ranker_v1 import fit_expanding_lgbm_predictions
from research_qlib_ridge_topk_v1 import (
    BASELINE_WEIGHTS,
    baseline_scores,
    normalized_curve,
)
from app.lgbm_strategy import WINNER_CONFIG


DEFAULT_WARMUP_START = date(2024, 4, 24)
DEFAULT_START = date(2024, 9, 21)
DEFAULT_END = date(2026, 9, 18)
TOP50_SIZE = 50
REBALANCE_DAYS = 10
N_DROP = 1
COST_BPS = 12.0
BARRA_WEIGHT = 0.25

DATA_DIR = BASE_DIR / "data" / "cache" / "market_history_2y"
BACKTEST_DIR = DATA_DIR / "backtest"
METADATA_PATH = DATA_DIR / "metadata.json"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

OUTPUT_JSON = BASE_DIR / "data" / "dynamic_top50_lgbm_2024_2026.json"
OUTPUT_REPORT = BASE_DIR / "reports" / "dynamic_top50_lgbm_2024_2026.md"
OUTPUT_CURVE = BASE_DIR / "reports" / "dynamic_top50_lgbm_2024_2026_curve.csv"
OUTPUT_SVG = BASE_DIR / "reports" / "dynamic_top50_lgbm_2024_2026.svg"
OUTPUT_TOP50 = BASE_DIR / "reports" / "dynamic_top50_lgbm_2024_2026_top50.csv"
OUTPUT_INTERVALS = BASE_DIR / "reports" / "dynamic_top50_lgbm_2024_2026_membership_intervals.csv"
OUTPUT_REBALANCES = BASE_DIR / "reports" / "dynamic_top50_lgbm_2024_2026_rebalances.csv"

RAW_COLUMNS = [
    "date",
    "code",
    "name",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "pct_change",
    "tradestatus",
    "is_st",
    "pe",
    "pb",
    "source",
]


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def _canonical_code(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def _parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def load_full_market_history(
    data_dir: Path,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, dict]:
    """Load every successfully collected symbol, including non-selected stocks."""
    manifest_path = data_dir / "manifest.csv"
    backtest_dir = data_dir / "backtest"
    if not manifest_path.exists():
        raise RuntimeError(f"市场历史 manifest 不存在: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype={"code": str})
    manifest["code"] = manifest["code"].map(_canonical_code)
    downloaded = manifest[manifest["status"].eq("downloaded")].copy()
    if downloaded.empty:
        raise RuntimeError("manifest 中没有可用历史股票")

    frames = []
    missing_files = []
    for row in downloaded.itertuples(index=False):
        path = ROOT_DIR / str(row.cache_file)
        if not path.exists():
            path = backtest_dir / f"{row.code}_*_qfq.csv"
            matches = sorted(path.parent.glob(path.name)) if "*" in path.name else []
            path = matches[0] if matches else None
        if path is None or not path.exists():
            missing_files.append(row.code)
            continue
        frame = pd.read_csv(path, usecols=lambda column: column in RAW_COLUMNS)
        if frame.empty:
            missing_files.append(row.code)
            continue
        frame["code"] = frame["code"].map(_canonical_code).fillna(row.code)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame[frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        if frame.empty:
            missing_files.append(row.code)
            continue
        for column in set(RAW_COLUMNS) - {"date", "code", "name", "source"}:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(frame)

    if not frames:
        raise RuntimeError("可用历史文件全部为空")
    history = pd.concat(frames, ignore_index=True)
    history = (
        history.sort_values(["code", "date"], kind="mergesort")
        .drop_duplicates(["date", "code"], keep="last")
        .reset_index(drop=True)
    )
    history["code"] = history["code"].map(_canonical_code)
    history["available_date"] = history["date"]
    history["universe_member"] = False
    history["universe_selection_date"] = pd.NaT
    history["universe_rank"] = np.nan
    history["universe_selection_score"] = np.nan
    metadata = {}
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "manifest_downloaded_codes": int(len(downloaded)),
        "loaded_codes": int(history["code"].nunique()),
        "loaded_rows": int(len(history)),
        "missing_files_in_manifest": sorted(set(missing_files)),
        "loaded_first_date": history["date"].min().date().isoformat(),
        "loaded_last_date": history["date"].max().date().isoformat(),
    })
    return history, metadata


def historical_selection_dates(
    dates: pd.DatetimeIndex,
    formal_start: date,
    step: int = REBALANCE_DAYS,
) -> pd.DatetimeIndex:
    """Align historical universe selections with the strategy's signal grid."""
    if step < 1:
        raise ValueError("selection step must be positive")
    ordered = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    first_formal = int(np.searchsorted(ordered.values, np.datetime64(formal_start)))
    if first_formal >= len(ordered):
        raise ValueError("formal backtest start is after the available history")
    indices = list(range(first_formal, -1, -step))
    indices.extend(range(first_formal + step, len(ordered), step))
    return ordered[sorted(set(indices))]


def _selector_cross_by_date(selector_frame: pd.DataFrame):
    return {
        pd.Timestamp(trade_date): group
        for trade_date, group in selector_frame.groupby("date", sort=False)
    }


def select_historical_top50(
    selector_frame: pd.DataFrame,
    selection_dates: pd.DatetimeIndex,
    top_n: int = TOP50_SIZE,
) -> tuple[pd.DataFrame, list[dict]]:
    """Select a new Top50 using only the cross-section visible on each date."""
    engine = BacktestEngine()
    grouped = _selector_cross_by_date(selector_frame)
    top_rows = []
    audits = []
    for selection_date in selection_dates:
        cross = grouped.get(pd.Timestamp(selection_date))
        if cross is None:
            audits.append({
                "selection_date": pd.Timestamp(selection_date).date().isoformat(),
                "eligible_count": 0,
                "selected_count": 0,
                "missing_to_top50": top_n,
                "status": "missing_market_date",
            })
            continue
        eligible = cross[
            cross["tradestatus"].eq(1)
            & cross["is_st"].eq(0)
            & cross["close"].gt(1)
            & cross["amount_20"].ge(10_000_000)
        ].dropna(subset=[
            "ret_5",
            "ret_20",
            "ret_60",
            "volatility_20",
            "volatility_60",
            "amount_20",
            "amount_60",
            "absolute_return_60",
            "ma_20",
        ])
        if eligible.empty:
            audits.append({
                "selection_date": pd.Timestamp(selection_date).date().isoformat(),
                "eligible_count": 0,
                "selected_count": 0,
                "missing_to_top50": top_n,
                "status": "no_eligible_cross_section",
            })
            continue
        scored = engine._score(eligible.copy(), BASELINE_WEIGHTS).sort_values(
            ["score", "amount_20", "code"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        selected = scored.head(top_n).copy()
        selected["selection_date"] = pd.Timestamp(selection_date)
        selected["available_date"] = pd.Timestamp(selection_date)
        selected["universe_rank"] = np.arange(1, len(selected) + 1)
        selected["universe_selection_score"] = selected["score"].astype(float)
        top_rows.append(selected[[
            "selection_date",
            "available_date",
            "code",
            "name",
            "universe_rank",
            "universe_selection_score",
        ]])
        audits.append({
            "selection_date": pd.Timestamp(selection_date).date().isoformat(),
            "eligible_count": int(len(scored)),
            "selected_count": int(len(selected)),
            "missing_to_top50": int(max(0, top_n - len(selected))),
            "selection_score_min": float(selected["score"].min()) if not selected.empty else None,
            "selection_score_max": float(selected["score"].max()) if not selected.empty else None,
            "status": "ok" if len(selected) == top_n else "partial_top50",
        })
    if not top_rows:
        raise RuntimeError("历史日期上没有产生任何 Top50")
    top50 = pd.concat(top_rows, ignore_index=True)
    top50["code"] = top50["code"].map(_canonical_code)
    top50["selection_date"] = pd.to_datetime(top50["selection_date"])
    top50["available_date"] = pd.to_datetime(top50["available_date"])
    return top50, audits


def attach_historical_membership(
    history: pd.DataFrame,
    top50: pd.DataFrame,
    selection_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Attach the active historical Top50 to every available market row."""
    output = history.copy()
    output["date"] = pd.to_datetime(output["date"])
    output["code"] = output["code"].map(_canonical_code)
    selection_dates = pd.DatetimeIndex(selection_dates)
    asof_selection = pd.Series(
        selection_dates,
        index=selection_dates,
        dtype="datetime64[ns]",
    )
    output["universe_selection_date"] = (
        asof_selection.reindex(output["date"], method="ffill")
        .set_axis(output.index)
        .to_numpy()
    )
    output["universe_member"] = False
    output["universe_rank"] = np.nan
    output["universe_selection_score"] = np.nan
    membership = top50.rename(columns={
        "selection_date": "universe_selection_date",
        "universe_rank": "selected_rank",
        "universe_selection_score": "selected_score",
    })[[
        "universe_selection_date",
        "code",
        "selected_rank",
        "selected_score",
    ]]
    output = output.merge(
        membership,
        on=["universe_selection_date", "code"],
        how="left",
        validate="many_to_one",
    )
    output["universe_member"] = output["selected_rank"].notna()
    output["universe_rank"] = output["selected_rank"]
    output["universe_selection_score"] = output["selected_score"]
    output = output.drop(columns=["selected_rank", "selected_score"])
    output["available_date"] = output["date"]
    return output.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)


def build_membership_intervals(
    top50: pd.DataFrame,
    selection_dates: pd.DatetimeIndex,
    all_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Convert period snapshots into auditable entry and exit intervals."""
    selection_dates = pd.DatetimeIndex(selection_dates)
    selection_slot = {
        pd.Timestamp(value): index for index, value in enumerate(selection_dates)
    }
    all_position = {
        pd.Timestamp(value): index for index, value in enumerate(all_dates)
    }
    previous_date = {
        pd.Timestamp(selection_date): (
            pd.Timestamp(all_dates[all_position[pd.Timestamp(selection_date)] - 1])
            if all_position[pd.Timestamp(selection_date)] > 0
            else pd.Timestamp(selection_date)
        )
        for selection_date in selection_dates
    }
    rows = []
    for code, group in top50.groupby("code", sort=True):
        ordered = group.sort_values("selection_date")
        current = None
        for row in ordered.itertuples(index=False):
            selection_date = pd.Timestamp(row.selection_date)
            slot = selection_slot[selection_date]
            if current is None or slot != current["last_slot"] + 1:
                if current is not None:
                    boundary_slot = min(current["last_slot"] + 1, len(selection_dates) - 1)
                    boundary_date = pd.Timestamp(selection_dates[boundary_slot])
                    boundary_index = max(all_position[boundary_date] - 1, 0)
                    current["exit_date"] = pd.Timestamp(
                        all_dates[boundary_index]
                    ).date().isoformat()
                    rows.append(current)
                current = {
                    "code": _canonical_code(code),
                    "entry_date": selection_date.date().isoformat(),
                    "exit_date": previous_date[selection_date].date().isoformat(),
                    "selection_periods": [selection_date.date().isoformat()],
                    "entry_rank": int(row.universe_rank),
                    "exit_rank": int(row.universe_rank),
                    "last_slot": slot,
                }
            else:
                current["exit_date"] = previous_date[selection_date].date().isoformat()
                current["selection_periods"].append(selection_date.date().isoformat())
                current["exit_rank"] = int(row.universe_rank)
                current["last_slot"] = slot
        if current is not None:
            final_slot = current.pop("last_slot")
            final_selection_date = pd.Timestamp(selection_dates[final_slot])
            end_index = min(
                all_position[final_selection_date] + REBALANCE_DAYS - 1,
                len(all_dates) - 1,
            )
            current["exit_date"] = pd.Timestamp(all_dates[end_index]).date().isoformat()
            rows.append(current)
    return pd.DataFrame(rows, columns=[
        "code",
        "entry_date",
        "exit_date",
        "selection_periods",
        "entry_rank",
        "exit_rank",
    ])


def _member_mask(frame: pd.DataFrame) -> pd.Series:
    if "universe_member" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return frame["universe_member"].fillna(False).astype(bool)


def blend_model_predictions_historical(
    frame: pd.DataFrame,
    baseline_predictions: pd.Series,
    barra_predictions: pd.Series,
    barra_weight: float,
) -> pd.Series:
    """Blend model branches by ranks inside the active historical Top50."""
    if not 0.0 <= barra_weight <= 1.0:
        raise ValueError("Barra 融合权重必须在 [0, 1]")
    member = _member_mask(frame)
    dates = frame["date"]
    baseline_rank = pd.Series(np.nan, index=frame.index, dtype=float)
    barra_rank = pd.Series(np.nan, index=frame.index, dtype=float)
    baseline_rank.loc[member] = baseline_predictions.loc[member].groupby(
        dates.loc[member]
    ).rank(pct=True)
    barra_rank.loc[member] = barra_predictions.loc[member].groupby(
        dates.loc[member]
    ).rank(pct=True)
    return baseline_rank.mul(1.0 - barra_weight).add(
        barra_rank.mul(barra_weight)
    )


def blend_scores_historical(
    frame: pd.DataFrame,
    model_prediction: pd.Series,
    baseline_prediction: pd.Series,
    baseline_weight: float,
) -> pd.DataFrame:
    """Apply the old 60/40 blend, ranking only inside each historical Top50."""
    if not 0.0 <= baseline_weight <= 1.0:
        raise ValueError("规则分支权重必须在 [0, 1]")
    score_columns = [
        "date",
        "code",
        "tradestatus",
        "is_st",
        "close",
        "amount_20",
        "universe_member",
        "universe_selection_date",
        "available_date",
        "universe_rank",
    ]
    score_columns = [column for column in score_columns if column in frame.columns]
    scores = frame[score_columns].copy()
    member = _member_mask(frame)
    eligible = (
        member
        & frame["tradestatus"].eq(1)
        & frame["is_st"].eq(0)
        & frame["close"].gt(1)
        & frame["amount_20"].ge(10_000_000)
    )
    model_rank = pd.Series(np.nan, index=frame.index, dtype=float)
    model_rank.loc[eligible] = model_prediction.loc[eligible].groupby(
        frame.loc[eligible, "date"]
    ).rank(pct=True)
    scores["score"] = model_rank.mul(1.0 - baseline_weight).add(
        baseline_prediction.mul(baseline_weight),
        fill_value=0.0,
    )
    scores.loc[~eligible, "score"] = np.nan
    return scores


def _metrics_from_curve(curve: pd.Series) -> dict:
    clean = curve.dropna().astype(float)
    if clean.empty:
        return {}
    returns = clean.pct_change().dropna()
    years = max((clean.index[-1] - clean.index[0]).days / 365.2425, 1 / 365.2425)
    volatility = float(returns.std(ddof=0) * np.sqrt(252.0)) if not returns.empty else 0.0
    total = float(clean.iloc[-1] / clean.iloc[0] - 1.0)
    return {
        "start_date": clean.index[0].date().isoformat(),
        "end_date": clean.index[-1].date().isoformat(),
        "observations": int(len(clean)),
        "terminal_value": float(clean.iloc[-1] / clean.iloc[0]),
        "total_return": total,
        "annual_return": float((1.0 + total) ** (1.0 / years) - 1.0),
        "annual_volatility": volatility,
        "sharpe_zero_rate": float(returns.mean() * 252.0 / volatility) if volatility > 0 else 0.0,
        "max_drawdown": float(clean.div(clean.cummax()).sub(1.0).min()),
    }


def run_strategy(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    start: date,
    end: date,
    label: str,
    feature_sets: dict[str, list[str]],
) -> tuple[dict, pd.Series, dict]:
    print(f"training {label}: Alpha158-lite monthly expanding branch", flush=True)
    base_prediction, base_audit = fit_expanding_lgbm_predictions(
        frame,
        feature_sets["alpha158_baseline"],
        WINNER_CONFIG,
        start,
        end,
    )
    print(f"training {label}: Alpha158+Barra monthly expanding branch", flush=True)
    barra_prediction, barra_audit = fit_expanding_lgbm_predictions(
        frame,
        feature_sets["alpha158_plus_barra"],
        WINNER_CONFIG,
        start,
        end,
    )
    prediction = blend_model_predictions_historical(
        frame, base_prediction, barra_prediction, BARRA_WEIGHT
    )
    rules = baseline_scores(frame, start, end)
    scores = blend_scores_historical(
        frame,
        prediction,
        rules,
        WINNER_CONFIG["baseline_weight"],
    )
    scores["expected_return"] = 0.0
    scores["downside_probability"] = 0.0
    print(f"running {label}: common next-open portfolio replay", flush=True)
    result = multihead_portfolio_backtest(
        history,
        scores,
        {
            "return_weight": 0.0,
            "safety_weight": 0.0,
            "use_hurdle": False,
            "max_k": 5,
            "min_k": 5,
            "risk_budget": None,
            "minimum_exposure": 1.0,
        },
        start,
        end,
        rebalance_days=REBALANCE_DAYS,
        n_drop=N_DROP,
        cost_bps=COST_BPS,
    )
    curve = normalized_curve(result, "strategy_return")
    curve.index = pd.to_datetime(curve.index)
    return result, curve, {
        "label": label,
        "base_audit": base_audit,
        "barra_audit": barra_audit,
        "strict_training_checks": {
            "alpha158_baseline": {
                row["month"]: pd.Timestamp(row["max_training_label_end_date"])
                < pd.Timestamp(row["prediction_start"])
                for row in base_audit.get("maturity_audit", [])
            },
            "alpha158_plus_barra": {
                row["month"]: pd.Timestamp(row["max_training_label_end_date"])
                < pd.Timestamp(row["prediction_start"])
                for row in barra_audit.get("maturity_audit", [])
            },
        },
    }


def _clean_result(result: dict) -> dict:
    return {
        "metrics": result["metrics"],
        "return_periods": result["return_periods"],
        "rebalances": result["rebalances"],
        "config": result["config"],
        "exposure_summary": {
            "average": float(result["exposure"].mean()),
            "minimum": float(result["exposure"].min()),
            "cash_days": int(result["exposure"].lt(0.01).sum()),
        },
    }


def _coverage_audit(history: pd.DataFrame, start: date, end: date) -> dict:
    dates = pd.DatetimeIndex(sorted(history["date"].unique()))
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    member = history[history["universe_member"] & history["date"].isin(dates)]
    counts = member.groupby("date")["code"].nunique().reindex(dates, fill_value=0)
    tradeable = member[
        member["tradestatus"].eq(1)
        & member["is_st"].eq(0)
        & member["close"].gt(1)
        & member["open"].gt(0)
    ].groupby("date")["code"].nunique().reindex(dates, fill_value=0)
    missing_rows = int((counts < TOP50_SIZE).sum())
    return {
        "formal_trading_days": int(len(dates)),
        "member_count_min": int(counts.min()) if not counts.empty else 0,
        "member_count_median": float(counts.median()) if not counts.empty else 0.0,
        "member_count_max": int(counts.max()) if not counts.empty else 0,
        "tradeable_member_count_min": int(tradeable.min()) if not tradeable.empty else 0,
        "tradeable_member_count_median": float(tradeable.median()) if not tradeable.empty else 0.0,
        "tradeable_member_count_max": int(tradeable.max()) if not tradeable.empty else 0,
        "days_below_top50_loaded_rows": missing_rows,
        "coverage_rate_loaded_rows": float((counts >= TOP50_SIZE).mean()) if not counts.empty else 0.0,
    }


def _strict_audit(
    history: pd.DataFrame,
    top50: pd.DataFrame,
    selection_dates: pd.DatetimeIndex,
    dynamic_details: dict,
    dynamic_result: dict,
) -> dict:
    member = history[history["universe_member"]].copy()
    selection_asof = (
        pd.to_datetime(member["universe_selection_date"], errors="coerce")
        <= pd.to_datetime(member["date"], errors="coerce")
    ).all()
    available_asof = (
        pd.to_datetime(member["available_date"], errors="coerce")
        <= pd.to_datetime(member["date"], errors="coerce")
    ).all()
    top50_dates = pd.to_datetime(top50["selection_date"], errors="coerce")
    selection_grid_ok = bool(set(top50_dates.dropna().unique()).issubset(set(selection_dates)))
    training_checks = dynamic_details["strict_training_checks"]
    training_mature = all(
        checks and all(checks.values()) for checks in training_checks.values()
    )
    signal_execution = all(
        pd.Timestamp(row["signal_date"]) < pd.Timestamp(row["execution_date"])
        for row in dynamic_result.get("rebalances", [])
    )
    return {
        "top50_selection_only_uses_asof_rows": bool(selection_asof),
        "top50_available_date_not_after_observation": bool(available_asof),
        "selection_dates_on_declared_grid": selection_grid_ok,
        "all_training_labels_mature_before_month_refit": bool(training_mature),
        "signal_close_before_next_open_execution": bool(signal_execution),
        "overall_dynamic_no_lookahead_checks": bool(
            selection_asof
            and available_asof
            and selection_grid_ok
            and training_mature
            and signal_execution
        ),
        "strict_point_in_time_full_market": False,
    }


def render_svg(curves: pd.DataFrame, metrics: dict) -> None:
    width, height = 1400, 800
    left, right, top, bottom = 100, 330, 105, 85
    plot_width, plot_height = width - left - right, height - top - bottom
    columns = ["dynamic_top50_lgbm", "frozen_current_top50", "market_benchmark"]
    labels = {
        "dynamic_top50_lgbm": "Dynamic historical Top50 LGBM",
        "frozen_current_top50": "Frozen current Top50",
        "market_benchmark": "Full-market equal weight",
    }
    colors = {
        "dynamic_top50_lgbm": "#176B87",
        "frozen_current_top50": "#C2413B",
        "market_benchmark": "#667085",
    }
    values = curves[columns].to_numpy(dtype=float)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    padding = max((high - low) * 0.06, 0.05)
    low = max(0.0, low - padding)
    high += padding

    def x(index: int) -> float:
        return left + plot_width * index / max(len(curves) - 1, 1)

    def y(value: float) -> float:
        return top + plot_height * (high - value) / max(high - low, 1e-12)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAFAF7"/>',
        '<text x="100" y="34" font-family="Segoe UI, Arial" font-size="24" font-weight="600" fill="#1F2933">Historical dynamic Top50 LGBM backtest</text>',
        f'<text x="100" y="64" font-family="Segoe UI, Arial" font-size="13" fill="#52606D">{curves.index[0].date()} to {curves.index[-1].date()} | all curves start at 1.00 | T-close to T+1 open</text>',
        f'<text x="100" y="86" font-family="Segoe UI, Arial" font-size="13" fill="#52606D">Dynamic CAGR {metrics["dynamic_top50_lgbm"]["annual_return"]:.2%} | Frozen CAGR {metrics["frozen_current_top50"]["annual_return"]:.2%} | cost 12bp one-way</text>',
    ]
    for tick in np.linspace(low, high, 6):
        yy = y(float(tick))
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_width}" y2="{yy:.2f}" stroke="#D9E2E8" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{yy + 4:.2f}" text-anchor="end" font-family="Segoe UI, Arial" font-size="12" fill="#52606D">{tick:.1f}x</text>')
    dates = pd.DatetimeIndex(curves.index)
    for year in range(dates.min().year, dates.max().year + 1):
        index = int(np.argmin(np.abs((dates - pd.Timestamp(year, 1, 1)).days)))
        parts.append(f'<text x="{x(index):.2f}" y="{height - 42}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="12" fill="#52606D">{year}</text>')
    for column in columns:
        points = " ".join(
            f"{x(index):.2f},{y(float(value)):.2f}"
            for index, value in enumerate(curves[column])
        )
        value = float(curves[column].iloc[-1])
        yy = y(value)
        parts.append(f'<polyline data-series="{column}" points="{points}" fill="none" stroke="{colors[column]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{left + plot_width}" cy="{yy:.2f}" r="5" fill="{colors[column]}"/>')
        parts.append(f'<text x="{left + plot_width + 18}" y="{yy + 5:.2f}" font-family="Segoe UI, Arial" font-size="13" fill="#1F2933">{labels[column]} {value:.2f}x</text>')
    parts.extend([
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#CBD5DC" stroke-width="1"/>',
        f'<text x="{left + plot_width / 2:.2f}" y="{height - 10}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="13" fill="#52606D">Calendar date</text>',
        f'<text x="25" y="{top + plot_height / 2:.2f}" transform="rotate(-90 25 {top + plot_height / 2:.2f})" text-anchor="middle" font-family="Segoe UI, Arial" font-size="13" fill="#52606D">Normalized net value (x)</text>',
        '</svg>',
    ])
    OUTPUT_SVG.write_text("\n".join(parts), encoding="utf-8")


def _write_report(payload: dict) -> None:
    dynamic = payload["results"]["dynamic_top50_lgbm"]["metrics"]
    frozen = payload["results"]["frozen_current_top50"]["metrics"]
    lines = [
        "# Historical Dynamic Top50 LGBM Backtest",
        "",
        "本次回测把全市场可用历史数据先用于每个历史调仓日的 Top50 选取，再执行原有 LGBM 组合策略。它不把 2026 年的当前 Top50 回填到 2024-2025 年。",
        "",
        "## 回测口径",
        "",
        f"- 数据区间：预热 {payload['window']['warmup_start']} 至 {payload['window']['end']}；正式回测 {payload['window']['start']} 至 {payload['window']['end']}。",
        f"- 全市场输入：{payload['market_data']['loaded_codes']} 只可用股票，正式区间 {payload['market_data']['formal_trading_days']} 个交易日。",
        "- Top50 选择：每 10 个交易日按当日可见的旧策略规则因子重新排序，取前 50；全市场参与选取，模型训练和排名只使用曾经进入历史 Top50 的股票，并在每个日期再限制为当期 Top50。",
        "- LGBM：75% Alpha158-lite + 25% Alpha158+Barra；模型排名与规则排名按 60/40 融合；月度 expanding LambdaRank。",
        "- 执行：收盘生成信号，下一交易日开盘成交，Top5，10 个交易日调仓，最多替换 1 只，单边成本 12bp。",
        "",
        "## 结果",
        "",
        "| 组合 | 终值 | 总收益 | 年化 | 夏普 | 最大回撤 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 历史动态 Top50 LGBM | {dynamic['terminal_value']:.2f}x | {dynamic['total_return']:.2%} | {dynamic['annual_return']:.2%} | {dynamic['sharpe_zero_rate']:.3f} | {dynamic['max_drawdown']:.2%} |",
        f"| 当前冻结 Top50 对照 | {frozen['terminal_value']:.2f}x | {frozen['total_return']:.2%} | {frozen['annual_return']:.2%} | {frozen['sharpe_zero_rate']:.3f} | {frozen['max_drawdown']:.2%} |",
        "",
        "## 严格检查",
        "",
    ]
    for key, value in payload["strict_audit"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "## 股票池覆盖",
        "",
        f"- 选股快照期数：{payload['universe']['selection_periods']}；Top50 记录数：{payload['universe']['top50_rows']}。",
        f"- 正式区间每日已加载 Top50 覆盖率：{payload['universe']['coverage']['coverage_rate_loaded_rows']:.2%}；每日成员数范围：{payload['universe']['coverage']['member_count_min']} / {payload['universe']['coverage']['member_count_median']:.0f} / {payload['universe']['coverage']['member_count_max']}。",
        f"- 成员区间文件：`{OUTPUT_INTERVALS.relative_to(BASE_DIR)}`；每期 Top50 文件：`{OUTPUT_TOP50.relative_to(BASE_DIR)}`。",
        "",
        "## 限制",
        "",
        "- 该结果的 Top50 选择和模型训练顺序已按历史日期隔离，但数据集仍来自 2026-09-18 的当前股票代码快照，未包含历史退市股票。",
        "- 价格是当前版本前复权 qfq，不是严格 as-of 复权价；历史每日 ST、停牌、涨跌停和可交易状态没有完整回溯。",
        "- 因此结果标签是 `dynamic_top50_historical_observed_universe`，不是 `strict_point_in_time_full_market`，不能直接当作无偏生产收益承诺。",
        "",
        "## 文件",
        "",
        f"- JSON：`{OUTPUT_JSON.relative_to(BASE_DIR)}`",
        f"- 曲线：`{OUTPUT_CURVE.relative_to(BASE_DIR)}`",
        f"- 图：`{OUTPUT_SVG.relative_to(BASE_DIR)}`",
    ])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="历史动态 Top50 LGBM 两年回测")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--warmup-start", default=DEFAULT_WARMUP_START.isoformat())
    parser.add_argument("--start", default=DEFAULT_START.isoformat())
    parser.add_argument("--end", default=DEFAULT_END.isoformat())
    parser.add_argument("--skip-frozen", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    warmup_start = _parse_date(args.warmup_start)
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if not warmup_start < start <= end:
        raise ValueError("日期必须满足 warmup_start < start <= end")

    history, market_metadata = load_full_market_history(args.data_dir, warmup_start, end)
    all_dates = pd.DatetimeIndex(sorted(history["date"].unique()))
    formal_dates = all_dates[(all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))]
    selection_dates = historical_selection_dates(all_dates, start, REBALANCE_DAYS)
    print(
        f"loaded full market: {history['code'].nunique()} codes, {len(history)} rows, "
        f"{len(formal_dates)} formal dates",
        flush=True,
    )
    print(f"selecting historical Top50 on {len(selection_dates)} signal-grid dates", flush=True)

    selector_engine = BacktestEngine()
    selector_frame = selector_engine._features(history.copy())
    top50, selection_audits = select_historical_top50(
        selector_frame, selection_dates, TOP50_SIZE
    )
    del selector_frame
    history = attach_historical_membership(history, top50, selection_dates)
    model_codes = sorted(top50["code"].unique().tolist())
    model_history = history[history["code"].isin(model_codes)].copy()
    model_rows = int(len(model_history))
    intervals = build_membership_intervals(top50, selection_dates, all_dates)
    coverage = _coverage_audit(history, start, end)
    for artifact in (
        OUTPUT_JSON,
        OUTPUT_REPORT,
        OUTPUT_CURVE,
        OUTPUT_SVG,
        OUTPUT_TOP50,
        OUTPUT_INTERVALS,
        OUTPUT_REBALANCES,
    ):
        artifact.parent.mkdir(parents=True, exist_ok=True)
    top50.to_csv(OUTPUT_TOP50, index=False, encoding="utf-8-sig")
    intervals_copy = intervals.copy()
    if not intervals_copy.empty:
        intervals_copy["selection_periods"] = intervals_copy["selection_periods"].map(
            lambda values: ",".join(values)
        )
    intervals_copy.to_csv(OUTPUT_INTERVALS, index=False, encoding="utf-8-sig")

    print(
        f"building Alpha158/Barra feature frame for historical Top50 union: {len(model_codes)} codes",
        flush=True,
    )
    frame, feature_sets = build_extended_feature_frame(model_history)
    del model_history
    frame = frame.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)

    dynamic_result, dynamic_curve, dynamic_details = run_strategy(
        frame, history, start, end, "dynamic historical Top50", feature_sets
    )

    frozen_codes = []
    frozen_details = None
    if not args.skip_frozen:
        source_path = BASE_DIR / "data" / "factor_optimization.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        frozen_candidates, frozen_snapshot_id = frozen_research_candidates(source["generated_at"])
        frozen_codes = [_canonical_code(item["code"]) for item in frozen_candidates[:TOP50_SIZE]]
        frozen_frame = frame[frame["code"].isin(frozen_codes)].copy()
        frozen_frame["universe_member"] = True
        frozen_frame["universe_selection_date"] = pd.Timestamp(start)
        frozen_frame["universe_rank"] = np.nan
        frozen_frame["universe_selection_score"] = np.nan
        if frozen_frame["code"].nunique() < 2:
            raise RuntimeError("当前冻结 Top50 在两年数据中不足两只")
        frozen_result, frozen_curve, frozen_details = run_strategy(
            frozen_frame, history, start, end, "frozen current Top50", feature_sets
        )
        frozen_details["snapshot_run_id"] = frozen_snapshot_id
    else:
        frozen_result = None
        frozen_curve = None

    curves = {
        "dynamic_top50_lgbm": dynamic_curve,
        "market_benchmark": normalized_curve(dynamic_result, "benchmark_return"),
    }
    metrics = {
        "dynamic_top50_lgbm": _metrics_from_curve(dynamic_curve),
        "market_benchmark": _metrics_from_curve(curves["market_benchmark"]),
    }
    if frozen_curve is not None:
        curves["frozen_current_top50"] = frozen_curve
        metrics["frozen_current_top50"] = _metrics_from_curve(frozen_curve)
    curve_frame = pd.concat(curves, axis=1, join="inner").sort_index()
    if curve_frame.empty or not np.allclose(curve_frame.iloc[0].to_numpy(dtype=float), 1.0):
        raise RuntimeError("动态、冻结和市场曲线无法对齐到共同归一化起点")
    curve_frame.index.name = "date"
    for column in list(curve_frame.columns):
        curve_frame[f"{column}_drawdown"] = curve_frame[column].div(
            curve_frame[column].cummax()
        ).sub(1.0)
    curve_frame.to_csv(OUTPUT_CURVE, encoding="utf-8-sig", float_format="%.10f")
    render_svg(curve_frame, metrics)

    strict_audit = _strict_audit(
        history,
        top50,
        selection_dates,
        dynamic_details,
        dynamic_result,
    )
    if not strict_audit["overall_dynamic_no_lookahead_checks"]:
        raise RuntimeError(f"动态 Top50 严格门禁未通过: {strict_audit}")

    source_counts = (
        history["source"].value_counts(dropna=False).to_dict()
        if "source" in history.columns
        else {}
    )
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "dynamic_top50_historical_observed_universe",
        "window": {
            "warmup_start": warmup_start.isoformat(),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "formal_trading_days": int(len(formal_dates)),
        },
        "strategy": {
            "architecture": "40% historical-universe rule rank + 60% monthly expanding LGBM; model = 75% Alpha158-lite + 25% Alpha158+Barra",
            "winner_config": WINNER_CONFIG,
            "barra_weight": BARRA_WEIGHT,
            "baseline_weights": BASELINE_WEIGHTS,
            "execution": "T-close signal; T+1 open; Top5; 10-session rebalance; max one replacement; 12bp one-way cost",
            "universe_selection": "full available market cross-section at each 10-session signal-grid date; fixed historical baseline factor score; top 50",
        },
        "market_data": {
            "dataset_label": "historical_observed_universe",
            "full_market_input_codes": int(market_metadata.get("manifest_downloaded_codes", 0)),
            "loaded_codes": int(history["code"].nunique()),
            "loaded_rows": int(len(history)),
            "model_feature_input_codes": int(len(model_codes)),
            "model_feature_input_rows": model_rows,
            "formal_trading_days": int(len(formal_dates)),
            "source_counts": source_counts,
            "metadata": market_metadata,
        },
        "universe": {
            "selection_frequency": "every_10_trading_sessions_aligned_to_formal_signal_grid",
            "selection_periods": int(len(selection_dates)),
            "top50_rows": int(len(top50)),
            "top50_unique_codes": int(top50["code"].nunique()),
            "partial_periods": int(sum(item["selected_count"] < TOP50_SIZE for item in selection_audits)),
            "selection_audit": selection_audits,
            "coverage": coverage,
            "intervals": int(len(intervals)),
            "frozen_snapshot_codes": int(len(frozen_codes)),
        },
        "training_audit": dynamic_details,
        "strict_audit": strict_audit,
        "results": {
            "dynamic_top50_lgbm": {
                "metrics": metrics["dynamic_top50_lgbm"],
                "backtest": _clean_result(dynamic_result),
            },
            "market_benchmark": {
                "metrics": metrics["market_benchmark"],
            },
        },
        "limitations": [
            "Top50 selection is historical/as-of within the observed 2026-09-18 current-code universe, not a historical delisting-complete universe.",
            "Current-vintage qfq prices are not strict as-of adjusted prices.",
            "Historical daily ST, suspension, limit-up/limit-down and tradability states are not fully reconstructed.",
            "The result is dynamic_top50_historical_observed_universe, not strict_point_in_time_full_market.",
        ],
        "artifacts": {
            "json": str(OUTPUT_JSON.relative_to(BASE_DIR)),
            "report": str(OUTPUT_REPORT.relative_to(BASE_DIR)),
            "curve": str(OUTPUT_CURVE.relative_to(BASE_DIR)),
            "svg": str(OUTPUT_SVG.relative_to(BASE_DIR)),
            "top50": str(OUTPUT_TOP50.relative_to(BASE_DIR)),
            "membership_intervals": str(OUTPUT_INTERVALS.relative_to(BASE_DIR)),
            "rebalances": str(OUTPUT_REBALANCES.relative_to(BASE_DIR)),
        },
    }
    if frozen_result is not None:
        payload["results"]["frozen_current_top50"] = {
            "metrics": metrics["frozen_current_top50"],
            "backtest": _clean_result(frozen_result),
            "details": frozen_details,
        }
    dynamic_rebalances = pd.DataFrame(dynamic_result["rebalances"])
    dynamic_rebalances.insert(0, "universe_mode", "dynamic_historical_top50")
    if frozen_result is not None:
        frozen_rebalances = pd.DataFrame(frozen_result["rebalances"])
        frozen_rebalances.insert(0, "universe_mode", "frozen_current_top50")
        rebalance_frame = pd.concat([dynamic_rebalances, frozen_rebalances], ignore_index=True)
    else:
        rebalance_frame = dynamic_rebalances
    rebalance_frame.to_csv(OUTPUT_REBALANCES, index=False, encoding="utf-8-sig")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_report(payload)
    print(json.dumps({
        "status": payload["status"],
        "window": payload["window"],
        "universe": payload["universe"],
        "strict_audit": payload["strict_audit"],
        "dynamic_metrics": payload["results"]["dynamic_top50_lgbm"]["metrics"],
        "frozen_metrics": payload["results"].get("frozen_current_top50", {}).get("metrics"),
        "artifacts": payload["artifacts"],
    }, ensure_ascii=False, indent=2, default=_json_default))

if __name__ == "__main__":
    main()
