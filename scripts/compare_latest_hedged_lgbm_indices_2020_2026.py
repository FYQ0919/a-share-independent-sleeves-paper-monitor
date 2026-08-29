from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STRATEGY_CURVE = ROOT / "reports" / "lgbm_csi300_hedge_v1_curves.csv"
STRATEGY_AUDIT = ROOT / "data" / "lgbm_csi300_hedge_v1.json"
INDEX_CURVE = ROOT / "reports" / "current_best_lgbm_vs_csi300_csi2000_2020_now.csv"
INDEX_AUDIT = ROOT / "data" / "current_best_lgbm_vs_csi300_csi2000_2020_now.json"
OUTPUT_JSON = ROOT / "data" / "latest_hedged_lgbm_vs_csi2000_csi300_2020_2026.json"
OUTPUT_CSV = ROOT / "reports" / "latest_hedged_lgbm_vs_csi2000_csi300_2020_2026.csv"
OUTPUT_REPORT = ROOT / "reports" / "latest_hedged_lgbm_vs_csi2000_csi300_2020_2026.md"
OUTPUT_VISUAL = Path(tempfile.gettempdir()) / "latest-lgbm-a-share-excess-2020-2026.html"


def normalize_common_curves(
    strategy: pd.Series, benchmarks: dict[str, pd.Series]
) -> pd.DataFrame:
    series = {"latest_hedged_lgbm": strategy, **benchmarks}
    common_dates: pd.DatetimeIndex | None = None
    for values in series.values():
        clean_index = pd.DatetimeIndex(values.dropna().index).drop_duplicates().sort_values()
        common_dates = (
            clean_index
            if common_dates is None
            else common_dates.intersection(clean_index).sort_values()
        )
    if common_dates is None or len(common_dates) < 2:
        raise RuntimeError("strategy and benchmarks have insufficient common dates")

    normalized = {}
    for name, values in series.items():
        clean = values[~values.index.duplicated(keep="last")].sort_index()
        aligned = clean.reindex(common_dates).astype(float)
        if aligned.isna().any() or not np.isfinite(aligned).all():
            raise RuntimeError(f"{name} contains invalid values on common dates")
        if float(aligned.iloc[0]) <= 0:
            raise RuntimeError(f"{name} has a non-positive starting value")
        normalized[name] = aligned.div(float(aligned.iloc[0]))

    frame = pd.DataFrame(normalized, index=common_dates)
    frame.index.name = "date"
    return frame


def curve_metrics(values: pd.Series) -> dict[str, float | str | int]:
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        raise ValueError("at least two curve observations are required")
    returns = clean.pct_change().dropna()
    elapsed_years = max(
        (clean.index[-1] - clean.index[0]).days / 365.2425,
        1.0 / 365.2425,
    )
    total_return = float(clean.iloc[-1] / clean.iloc[0] - 1.0)
    volatility = float(returns.std(ddof=0) * np.sqrt(252.0))
    return {
        "start_date": clean.index[0].date().isoformat(),
        "end_date": clean.index[-1].date().isoformat(),
        "observations": int(len(clean)),
        "elapsed_calendar_years": float(elapsed_years),
        "terminal_value": float(clean.iloc[-1] / clean.iloc[0]),
        "total_return": total_return,
        "annual_return": float((1.0 + total_return) ** (1.0 / elapsed_years) - 1.0),
        "annual_volatility": volatility,
        "sharpe_zero_rate": (
            float(returns.mean() * 252.0 / volatility) if volatility > 0 else 0.0
        ),
        "max_drawdown": float(clean.div(clean.cummax()).sub(1.0).min()),
    }


def excess_metrics(strategy: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    aligned = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1
    ).dropna()
    if len(aligned) < 3:
        raise ValueError("at least three aligned observations are required")
    daily = aligned.pct_change().dropna()
    active = daily["strategy"].sub(daily["benchmark"])
    tracking_error = float(active.std(ddof=0) * np.sqrt(252.0))
    benchmark_centered = daily["benchmark"].sub(daily["benchmark"].mean())
    strategy_centered = daily["strategy"].sub(daily["strategy"].mean())
    benchmark_variance = float(np.mean(np.square(benchmark_centered)))
    beta = (
        float(np.mean(strategy_centered * benchmark_centered) / benchmark_variance)
        if benchmark_variance > 0
        else 0.0
    )
    strategy_performance = curve_metrics(aligned["strategy"])
    benchmark_performance = curve_metrics(aligned["benchmark"])
    relative_wealth = float(aligned["strategy"].iloc[-1] / aligned["benchmark"].iloc[-1])
    return {
        "annualized_return_gap": float(
            strategy_performance["annual_return"]
            - benchmark_performance["annual_return"]
        ),
        "cumulative_return_arithmetic_gap": float(
            strategy_performance["total_return"]
            - benchmark_performance["total_return"]
        ),
        "relative_wealth": relative_wealth,
        "relative_cumulative_return": relative_wealth - 1.0,
        "tracking_error": tracking_error,
        "information_ratio": (
            float(active.mean() * 252.0 / tracking_error) if tracking_error > 0 else 0.0
        ),
        "beta": beta,
        "annualized_jensen_alpha_zero_rate": float(
            (daily["strategy"].mean() - beta * daily["benchmark"].mean()) * 252.0
        ),
    }


def load_inputs() -> tuple[pd.DataFrame, dict, dict]:
    strategy_audit = json.loads(STRATEGY_AUDIT.read_text(encoding="utf-8"))
    index_audit = json.loads(INDEX_AUDIT.read_text(encoding="utf-8"))
    if strategy_audit.get("status") != "historical_candidate_only":
        raise RuntimeError("latest hedge audit status is not historical_candidate_only")
    if not strategy_audit.get("promotion_gate", {}).get("passed"):
        raise RuntimeError("latest hedge candidate did not pass its historical gate")
    if strategy_audit.get("production_change") is not False:
        raise RuntimeError("comparison expects production configuration to remain unchanged")

    strategy_frame = pd.read_csv(STRATEGY_CURVE, parse_dates=["date"])
    index_frame = pd.read_csv(INDEX_CURVE, parse_dates=["date"])
    required_strategy = {"date", "lgbm_csi300_hedged"}
    required_indices = {"date", "csi2000", "csi300"}
    if not required_strategy.issubset(strategy_frame.columns):
        raise RuntimeError("latest hedge curve is missing required columns")
    if not required_indices.issubset(index_frame.columns):
        raise RuntimeError("index curve is missing required columns")

    strategy = strategy_frame.set_index("date")["lgbm_csi300_hedged"]
    indices = index_frame.set_index("date")[["csi2000", "csi300"]]
    curves = normalize_common_curves(
        strategy,
        {"csi2000": indices["csi2000"], "csi300": indices["csi300"]},
    )
    curves["relative_wealth_vs_csi2000"] = curves["latest_hedged_lgbm"].div(
        curves["csi2000"]
    )
    curves["relative_wealth_vs_csi300"] = curves["latest_hedged_lgbm"].div(
        curves["csi300"]
    )
    return curves, strategy_audit, index_audit


def write_inline_visual(curves: pd.DataFrame) -> None:
    sampled = curves.iloc[::5].copy()
    if sampled.index[-1] != curves.index[-1]:
        sampled = pd.concat([sampled, curves.iloc[[-1]]])
    chart_data = [
        {
            "d": date.date().isoformat(),
            "s": round(float(row["latest_hedged_lgbm"]), 6),
            "c2": round(float(row["csi2000"]), 6),
            "c3": round(float(row["csi300"]), 6),
            "r2": round(float(row["relative_wealth_vs_csi2000"]), 6),
            "r3": round(float(row["relative_wealth_vs_csi300"]), 6),
        }
        for date, row in sampled.iterrows()
    ]
    data_json = json.dumps(chart_data, ensure_ascii=True, separators=(",", ":"))
    fragment = f"""<div id="latest-lgbm-index-excess" class="latest-lgbm-index-excess">
  <h2>最新对冲 LGBM 与 A 股指数对比</h2>
  <div class="chart-caption text-muted">2020-01-02 至 2026-08-25；所有曲线起点归一化为 1.00</div>
  <div class="legend" data-legend="wealth" aria-label="Cumulative wealth series"></div>
  <div class="plot" data-plot="wealth"></div>
  <div class="legend" data-legend="relative" aria-label="Relative wealth series"></div>
  <div class="plot" data-plot="relative"></div>
  <div class="tooltip" role="tooltip" hidden></div>
</div>
<style>
  #latest-lgbm-index-excess {{ position: relative; width: 100%; color: var(--foreground); }}
  #latest-lgbm-index-excess h2 {{ margin: 0 0 4px; font-weight: 500; letter-spacing: 0; }}
  #latest-lgbm-index-excess .chart-caption {{ margin-bottom: 10px; }}
  #latest-lgbm-index-excess .legend {{ display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 6px 0 2px 64px; }}
  #latest-lgbm-index-excess .legend button {{ display: inline-flex; align-items: center; gap: 6px; padding: 3px 0; border: 0; background: transparent; color: var(--foreground); font: inherit; cursor: pointer; }}
  #latest-lgbm-index-excess .legend button[aria-pressed="false"] {{ color: var(--muted-foreground); }}
  #latest-lgbm-index-excess .swatch {{ width: 18px; height: 3px; background: var(--series-color); }}
  #latest-lgbm-index-excess .plot {{ width: 100%; }}
  #latest-lgbm-index-excess .plot + .legend {{ margin-top: 8px; }}
  #latest-lgbm-index-excess svg {{ display: block; width: 100%; color: var(--foreground); }}
  #latest-lgbm-index-excess .axis text {{ fill: var(--foreground); font-size: 12px; }}
  #latest-lgbm-index-excess .axis path, #latest-lgbm-index-excess .axis line {{ stroke: var(--border); }}
  #latest-lgbm-index-excess .grid line {{ stroke: var(--border); opacity: 0.45; }}
  #latest-lgbm-index-excess .axis-title {{ fill: var(--foreground); font-size: 12px; }}
  #latest-lgbm-index-excess .line {{ fill: none; stroke-width: 2; }}
  #latest-lgbm-index-excess .endpoint {{ fill: var(--foreground); font-size: 12px; }}
  #latest-lgbm-index-excess .hover-guide {{ stroke: var(--muted-foreground); stroke-width: 1; }}
  #latest-lgbm-index-excess .hover-marker {{ fill: var(--background); stroke-width: 2; }}
  #latest-lgbm-index-excess .tooltip {{ position: absolute; pointer-events: none; z-index: 10; padding: 8px 10px; border: 1px solid var(--border); background: var(--popover); color: var(--popover-foreground); font-size: 12px; }}
  @media (max-width: 420px) {{ #latest-lgbm-index-excess .legend {{ margin-left: 54px; gap: 2px 10px; }} }}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById("latest-lgbm-index-excess");
  const parseDate = d3.timeParse("%Y-%m-%d");
  const data = {data_json}.map(d => ({{...d, date: parseDate(d.d)}}));
  const tooltip = root.querySelector(".tooltip");
  const groups = {{
    wealth: [
      {{key: "s", label: "最新对冲 LGBM 7.06x", color: "var(--viz-series-1)", visible: true}},
      {{key: "c2", label: "中证2000 1.77x", color: "var(--viz-series-2)", visible: true}},
      {{key: "c3", label: "沪深300 1.10x", color: "var(--viz-series-3)", visible: true}}
    ],
    relative: [
      {{key: "r2", label: "相对中证2000 4.00x", color: "var(--viz-series-4)", visible: true}},
      {{key: "r3", label: "相对沪深300 6.44x", color: "var(--viz-series-5)", visible: true}}
    ]
  }};
  const configs = {{
    wealth: {{height: 330, yLabel: "累计净值（倍）", title: "累计净值"}},
    relative: {{height: 270, yLabel: "相对财富（倍）", title: "策略净值 / 指数净值"}}
  }};

  function buildLegend(name) {{
    const holder = d3.select(root.querySelector(`[data-legend="${{name}}"]`));
    holder.selectAll("button").data(groups[name]).join("button")
      .attr("type", "button")
      .attr("aria-pressed", d => String(d.visible))
      .on("click", function(event, d) {{
        d.visible = !d.visible;
        d3.select(this).attr("aria-pressed", String(d.visible));
        draw(name);
      }})
      .html(d => `<span class="swatch" style="--series-color:${{d.color}}"></span><span>${{d.label}}</span>`);
  }}

  function interpolatedValue(key, target) {{
    const bisect = d3.bisector(d => d.date).left;
    const i = Math.min(Math.max(bisect(data, target), 1), data.length - 1);
    const a = data[i - 1], b = data[i];
    const span = b.date - a.date;
    const t = span > 0 ? (target - a.date) / span : 0;
    return a[key] + (b[key] - a[key]) * t;
  }}

  function draw(name) {{
    const holder = root.querySelector(`[data-plot="${{name}}"]`);
    const config = configs[name];
    const width = Math.max(320, Math.floor(holder.getBoundingClientRect().width || 736));
    const height = config.height;
    const margin = {{top: 14, right: 28, bottom: 52, left: width < 430 ? 54 : 64}};
    const visible = groups[name].filter(d => d.visible);
    d3.select(holder).selectAll("svg").remove();
    const svg = d3.select(holder).append("svg")
      .attr("viewBox", `0 0 ${{width}} ${{height}}`)
      .attr("role", "img")
      .attr("aria-label", `${{config.title}} from January 2020 through August 2026`);
    svg.append("title").text(config.title);
    svg.append("desc").text("Interactive time series chart. Use the legend buttons to show or hide series.");
    const x = d3.scaleTime().domain(d3.extent(data, d => d.date)).range([margin.left, width - margin.right]);
    const rawExtent = d3.extent(visible.flatMap(s => data.map(d => d[s.key])));
    const fallbackExtent = name === "wealth" ? [0.8, 7.2] : [0.8, 6.6];
    const extent = rawExtent[0] == null ? fallbackExtent : rawExtent;
    const pad = Math.max((extent[1] - extent[0]) * 0.07, 0.08);
    const y = d3.scaleLinear().domain([Math.max(0, extent[0] - pad), extent[1] + pad]).nice().range([height - margin.bottom, margin.top]);
    const frameX = margin.left, frameY = margin.top, frameW = width - margin.left - margin.right, frameH = height - margin.top - margin.bottom;
    svg.append("rect").attr("data-chart-frame", "").attr("x", frameX).attr("y", frameY).attr("width", frameW).attr("height", frameH).attr("fill", "none").attr("stroke", "var(--border)");
    svg.append("g").attr("class", "grid").attr("transform", `translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-frameW).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${{height - margin.bottom}})`).call(d3.axisBottom(x).ticks(width < 430 ? 4 : 7).tickFormat(d3.timeFormat("%Y")));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d => `${{d.toFixed(1)}}x`));
    svg.append("text").attr("class", "axis-title").attr("data-axis", "x").attr("x", margin.left + frameW / 2).attr("y", height - 8).attr("text-anchor", "middle").text("日期");
    svg.append("text").attr("class", "axis-title").attr("data-axis", "y").attr("transform", `translate(15,${{margin.top + frameH / 2}}) rotate(-90)`).attr("text-anchor", "middle").text(config.yLabel);
    visible.forEach(series => {{
      const generator = d3.line().x(d => x(d.date)).y(d => y(d[series.key]));
      svg.append("path").datum(data).attr("class", "line").attr("data-series", series.key).attr("stroke", series.color).attr("d", generator);
      const last = data[data.length - 1];
      svg.append("circle").attr("cx", x(last.date)).attr("cy", y(last[series.key])).attr("r", 3).attr("fill", series.color);
    }});
    const guide = svg.append("line").attr("class", "hover-guide").attr("data-chart-hover-guide", "").attr("y1", margin.top).attr("y2", height - margin.bottom).attr("visibility", "hidden");
    const markers = svg.append("g");
    const overlay = svg.append("rect").attr("data-chart-hit", "").attr("data-chart-hover-overlay", "cross-series").attr("x", frameX).attr("y", frameY).attr("width", frameW).attr("height", frameH).attr("fill", "transparent").style("pointer-events", "all");
    overlay.on("pointermove", event => {{
      const [px] = d3.pointer(event, svg.node());
      const date = x.invert(Math.max(frameX, Math.min(frameX + frameW, px)));
      guide.attr("x1", x(date)).attr("x2", x(date)).attr("visibility", "visible");
      const rows = visible.map(series => ({{series, value: interpolatedValue(series.key, date)}}));
      markers.selectAll("circle").data(rows, d => d.series.key).join("circle")
        .attr("class", "hover-marker").attr("data-chart-hover-marker", "")
        .attr("cx", x(date)).attr("cy", d => y(d.value)).attr("r", 4).attr("stroke", d => d.series.color);
      tooltip.hidden = false;
      tooltip.innerHTML = `<strong>${{d3.timeFormat("%Y-%m-%d")(date)}}</strong>` + rows.map(d => `<div>${{d.series.label.split(" ").slice(0, -1).join(" ")}}: ${{d.value.toFixed(2)}}x</div>`).join("");
      const rootBox = root.getBoundingClientRect();
      const svgBox = svg.node().getBoundingClientRect();
      const left = Math.min(rootBox.width - 170, Math.max(4, svgBox.left - rootBox.left + event.offsetX + 12));
      tooltip.style.left = `${{left}}px`;
      tooltip.style.top = `${{svgBox.top - rootBox.top + 20}}px`;
    }}).on("pointerleave", () => {{ guide.attr("visibility", "hidden"); markers.selectAll("circle").remove(); tooltip.hidden = true; }});
  }}

  Object.keys(groups).forEach(buildLegend);
  Object.keys(groups).forEach(draw);
  const observer = new ResizeObserver(() => Object.keys(groups).forEach(draw));
  observer.observe(root);
}})();
</script>
"""
    OUTPUT_VISUAL.write_text(fragment, encoding="utf-8")


def main() -> None:
    curves, strategy_audit, index_audit = load_inputs()
    strategy_metrics = curve_metrics(curves["latest_hedged_lgbm"])
    benchmark_metrics = {
        name: curve_metrics(curves[name]) for name in ("csi2000", "csi300")
    }
    excess = {
        name: excess_metrics(curves["latest_hedged_lgbm"], curves[name])
        for name in ("csi2000", "csi300")
    }

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(OUTPUT_CSV, encoding="utf-8-sig", float_format="%.10f")
    write_inline_visual(curves)

    csi2000_source = index_audit["curves"]["csi2000"]
    csi300_source = index_audit["curves"]["csi300"]
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "revealed_historical_diagnostic_not_promoted",
        "window": {
            "requested": "2020-2026",
            "common_start": curves.index[0].date().isoformat(),
            "common_end": curves.index[-1].date().isoformat(),
            "common_trading_days": int(len(curves)),
            "elapsed_calendar_years": strategy_metrics["elapsed_calendar_years"],
        },
        "strategy": {
            "name": "75/25 LGBM plus causal CSI300 MA120 hedge",
            "parameters": strategy_audit["selected_parameters"],
            "selection_window": strategy_audit["selection_window"],
            "execution": strategy_audit["execution_audit"],
            "metrics": strategy_metrics,
        },
        "benchmarks": {
            "csi2000": {
                "name": "CSI2000",
                "symbol": "932000",
                "return_type": "price return; excludes dividends",
                "metrics": benchmark_metrics["csi2000"],
                "official_source": csi2000_source["official_source"],
                "proxy_source": csi2000_source["proxy_source"],
                "proxy_audit": csi2000_source["proxy_audit"],
            },
            "csi300": {
                "name": "CSI300",
                "symbol": "000300",
                "return_type": "price return; excludes dividends",
                "metrics": benchmark_metrics["csi300"],
                "source": csi300_source["source"],
            },
        },
        "excess": excess,
        "artifacts": {
            "json": str(OUTPUT_JSON.relative_to(ROOT)),
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(ROOT)),
            "inline_visual": str(OUTPUT_VISUAL),
        },
        "limitations": [
            "CSI2000 after 2025-12-31 is an ETF-calibrated proxy, not an official index observation.",
            "Both benchmarks are price-return indices and exclude dividends.",
            "The strategy hedge requires index futures or equivalent short exposure; basis, roll, margin and financing are not modeled.",
            "The current Top50 universe is backfilled and has constituent and survivorship bias.",
            "The 2020-2026 period has been repeatedly revealed and is diagnostic, not an untouched out-of-sample holdout.",
            "No production model, paper account or scheduler configuration was changed.",
        ],
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    labels = {"csi2000": "CSI2000", "csi300": "CSI300"}
    lines = [
        "# Latest hedged LGBM vs A-share indices, 2020-2026",
        "",
        f"- Common window: {payload['window']['common_start']} to {payload['window']['common_end']} ({len(curves)} trading days).",
        "- All curves start at 1.00; CAGR uses actual elapsed calendar years.",
        "- Status: historical diagnostic only; production and paper trading unchanged.",
        "",
        "| Curve | Terminal value | Total return | CAGR | Max drawdown | Sharpe (0% rate) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Latest hedged LGBM | {strategy_metrics['terminal_value']:.2f}x | {strategy_metrics['total_return']:.2%} | {strategy_metrics['annual_return']:.2%} | {strategy_metrics['max_drawdown']:.2%} | {strategy_metrics['sharpe_zero_rate']:.3f} |",
    ]
    for name in ("csi2000", "csi300"):
        metrics = benchmark_metrics[name]
        lines.append(
            f"| {labels[name]} | {metrics['terminal_value']:.2f}x | {metrics['total_return']:.2%} | {metrics['annual_return']:.2%} | {metrics['max_drawdown']:.2%} | {metrics['sharpe_zero_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Benchmark | CAGR gap | Arithmetic cumulative gap | Relative wealth | Relative cumulative return | Tracking error | Information ratio | Beta | Jensen alpha (0% rate) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("csi2000", "csi300"):
        metrics = excess[name]
        lines.append(
            f"| {labels[name]} | {metrics['annualized_return_gap']:+.2%} | {metrics['cumulative_return_arithmetic_gap']:+.2%} | {metrics['relative_wealth']:.2f}x | {metrics['relative_cumulative_return']:+.2%} | {metrics['tracking_error']:.2%} | {metrics['information_ratio']:.3f} | {metrics['beta']:.3f} | {metrics['annualized_jensen_alpha_zero_rate']:+.2%} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
