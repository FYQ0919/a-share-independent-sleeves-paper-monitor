from __future__ import annotations

from datetime import date, datetime
import html
import json
from pathlib import Path
import sys
import tempfile

import httpx
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.lgbm_strategy import WINNER_CONFIG
from compare_validation_nasdaq import frozen_research_candidates
from optimize_factor_strategy import load_current_history
from research_lgbm_multi_objective_v1 import (
    CANDIDATES as PORTFOLIO_CANDIDATES,
    multihead_portfolio_backtest,
)
from research_lgbm_new_factors_v1 import (
    blend_model_predictions,
    build_extended_feature_frame,
)
from research_qlib_lgbm_ranker_v1 import fit_expanding_lgbm_predictions
from research_qlib_ridge_topk_v1 import (
    SOURCE_OPTIMIZATION,
    baseline_scores,
    blend_scores,
    normalized_curve,
)


START = date(2016, 8, 25)
END = date(2026, 8, 28)
BARRA_WEIGHT = 0.25
REBALANCE_DAYS = 10
N_DROP = 1
COST_BPS = 12.0

OUTPUT_JSON = ROOT_DIR / "data" / "current_best_lgbm_vs_ndx_2016_2026.json"
OUTPUT_CSV = ROOT_DIR / "reports" / "current_best_lgbm_vs_ndx_2016_2026.csv"
OUTPUT_REPORT = ROOT_DIR / "reports" / "current_best_lgbm_vs_ndx_2016_2026.md"
OUTPUT_SVG = ROOT_DIR / "reports" / "current_best_lgbm_vs_ndx_2016_2026.svg"
OUTPUT_HTML = Path(tempfile.gettempdir()) / "current-best-lgbm-vs-ndx-2016-2026.html"
NDX_FALLBACK = ROOT_DIR / "reports" / "best_lgbm_vs_nasdaq100_10y.csv"
NDX_URL = (
    "https://api.nasdaq.com/api/quote/NDX/historical"
    "?assetclass=index&fromdate=2016-08-23&todate=2026-08-28&limit=4000"
)


def maturity_checks(audit: dict) -> dict[str, bool]:
    return {
        row["month"]: pd.Timestamp(row["max_training_label_end_date"])
        < pd.Timestamp(row["prediction_start"])
        for row in audit.get("maturity_audit", [])
    }


def curve_metrics(values: pd.Series) -> dict[str, float | str | int]:
    clean = values.dropna().astype(float)
    returns = clean.pct_change().dropna()
    elapsed_years = max((clean.index[-1] - clean.index[0]).days / 365.2425, 1 / 365.2425)
    volatility = float(returns.std(ddof=0) * np.sqrt(252.0))
    total_return = float(clean.iloc[-1] / clean.iloc[0] - 1.0)
    return {
        "start_date": clean.index[0].date().isoformat(),
        "end_date": clean.index[-1].date().isoformat(),
        "observations": int(len(clean)),
        "terminal_value": float(clean.iloc[-1] / clean.iloc[0]),
        "total_return": total_return,
        "annual_return": float((1.0 + total_return) ** (1.0 / elapsed_years) - 1.0),
        "annual_volatility": volatility,
        "sharpe_zero_rate": (
            float(returns.mean() * 252.0 / volatility) if volatility > 0 else 0.0
        ),
        "max_drawdown": float(clean.div(clean.cummax()).sub(1.0).min()),
    }


def yearly_returns(values: pd.Series) -> dict[str, float]:
    clean = values.dropna().astype(float)
    output: dict[str, float] = {}
    for year in sorted(clean.index.year.unique()):
        year_values = clean.loc[clean.index.year == year]
        before_year = clean.loc[clean.index < pd.Timestamp(int(year), 1, 1)]
        if year_values.empty:
            continue
        base = float(before_year.iloc[-1]) if not before_year.empty else float(year_values.iloc[0])
        output[str(year)] = float(year_values.iloc[-1] / base - 1.0)
    return output


def fetch_ndx() -> tuple[pd.Series, dict]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/",
    }
    try:
        with httpx.Client(headers=headers, timeout=30, trust_env=False) as client:
            response = client.get(NDX_URL)
            response.raise_for_status()
        rows = response.json().get("data", {}).get("tradesTable", {}).get("rows", [])
        if not rows:
            raise RuntimeError("Nasdaq API returned no NDX rows")
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y")
        frame["ndx"] = pd.to_numeric(
            frame["close"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )
        values = (
            frame.dropna(subset=["date", "ndx"])
            .drop_duplicates("date")
            .set_index("date")["ndx"]
            .sort_index()
        )
        source = {
            "mode": "live_nasdaq_api",
            "source": NDX_URL,
            "fetched_at": datetime.now().astimezone().isoformat(),
        }
    except Exception as exc:
        frame = pd.read_csv(NDX_FALLBACK, parse_dates=["date"])
        values = frame.set_index("date")["ndx"].sort_index().astype(float)
        values = values.loc[~values.index.weekday.isin([5, 6])]
        source = {
            "mode": "cached_fallback",
            "source": str(NDX_FALLBACK.relative_to(ROOT_DIR)),
            "fallback_reason": str(exc),
            "cache_modified_at": datetime.fromtimestamp(
                NDX_FALLBACK.stat().st_mtime
            ).astimezone().isoformat(),
        }
    values = values.loc[(values.index >= pd.Timestamp(START)) & (values.index <= pd.Timestamp(END))]
    if len(values) < 2:
        raise RuntimeError("NDX has insufficient observations in the requested window")
    return values, source


def build_common_curves(strategy: pd.Series, ndx: pd.Series) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    common_start = max(strategy.index.min(), ndx.index.min())
    common_end = min(strategy.index.max(), ndx.index.max())
    strategy = strategy.loc[(strategy.index >= common_start) & (strategy.index <= common_end)]
    ndx = ndx.loc[(ndx.index >= common_start) & (ndx.index <= common_end)]
    calendar = pd.date_range(common_start, common_end, freq="D")
    aligned = pd.concat(
        {
            "current_best_lgbm": strategy.reindex(calendar).ffill(),
            "nasdaq_100": ndx.reindex(calendar).ffill(),
        },
        axis=1,
    ).dropna()
    display = aligned.div(aligned.iloc[0])
    display.index.name = "date"
    if display.empty or not np.allclose(display.iloc[0].to_numpy(dtype=float), 1.0):
        raise RuntimeError("comparison curves do not share a normalized start")
    return display["current_best_lgbm"], display["nasdaq_100"], display


def render_svg(curve: pd.DataFrame, strategy_metrics: dict, ndx_metrics: dict) -> None:
    columns = ["current_best_lgbm", "nasdaq_100"]
    labels = {
        "current_best_lgbm": "Current best 75/25 LGBM",
        "nasdaq_100": "Nasdaq-100 (NDX)",
    }
    colors = {"current_best_lgbm": "#176B87", "nasdaq_100": "#C2413B"}
    width, height = 1400, 800
    left, right, top, bottom = 105, 325, 110, 92
    plot_width, plot_height = width - left - right, height - top - bottom
    values = curve[columns].to_numpy(dtype=float)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    padding = max((high - low) * 0.06, 0.05)
    low, high = max(0.0, low - padding), high + padding

    def x(index: int) -> float:
        return left + plot_width * index / max(len(curve) - 1, 1)

    def y(value: float) -> float:
        return top + plot_height * (high - value) / max(high - low, 1e-12)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAFAF7"/>',
        '<text x="105" y="35" font-family="Segoe UI, Arial" font-size="24" font-weight="600" fill="#1F2933">Current best LGBM vs Nasdaq-100, 2016-2026</text>',
        f'<text x="105" y="66" font-family="Segoe UI, Arial" font-size="13" fill="#52606D">Common window {curve.index[0].date()} to {curve.index[-1].date()} | Both start at 1.00 | NDX price return</text>',
        f'<text x="105" y="88" font-family="Segoe UI, Arial" font-size="13" fill="#52606D">CAGR: LGBM {strategy_metrics["annual_return"]:.2%} | NDX {ndx_metrics["annual_return"]:.2%} | LGBM includes 12 bps one-way cost</text>',
    ]
    for tick in np.linspace(low, high, 6):
        yy = y(float(tick))
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_width}" y2="{yy:.2f}" stroke="#D9E2E8" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{yy + 4:.2f}" text-anchor="end" font-family="Segoe UI, Arial" font-size="12" fill="#52606D">{tick:.1f}x</text>')
    dates = pd.DatetimeIndex(curve.index)
    for year in range(dates.min().year, dates.max().year + 1):
        index = int(np.argmin(np.abs((dates - pd.Timestamp(year, 1, 1)).days)))
        parts.append(f'<text x="{x(index):.2f}" y="{height - 48}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="12" fill="#52606D">{year}</text>')
    for column in columns:
        points = " ".join(
            f"{x(index):.2f},{y(float(value)):.2f}"
            for index, value in enumerate(curve[column])
        )
        value = float(curve[column].iloc[-1])
        yy = y(value)
        parts.append(f'<polyline data-series="{column}" points="{points}" fill="none" stroke="{colors[column]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{left + plot_width}" cy="{yy:.2f}" r="5" fill="{colors[column]}"/>')
        parts.append(f'<text x="{left + plot_width + 18}" y="{yy + 5:.2f}" font-family="Segoe UI, Arial" font-size="13" fill="#1F2933">{html.escape(labels[column])} {value:.2f}x</text>')
    parts.extend([
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#CBD5DC" stroke-width="1"/>',
        f'<text x="{left + plot_width / 2:.2f}" y="{height - 14}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="13" fill="#52606D">Calendar date</text>',
        f'<text x="25" y="{top + plot_height / 2:.2f}" transform="rotate(-90 25 {top + plot_height / 2:.2f})" text-anchor="middle" font-family="Segoe UI, Arial" font-size="13" fill="#52606D">Normalized net value (x)</text>',
        '</svg>',
    ])
    OUTPUT_SVG.write_text("\n".join(parts), encoding="utf-8")


def render_inline_html(curve: pd.DataFrame) -> None:
    sample_index = list(range(0, len(curve), 5))
    if sample_index[-1] != len(curve) - 1:
        sample_index.append(len(curve) - 1)
    sample = curve.iloc[sample_index].reset_index()
    sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
    data = sample.round(6).to_dict(orient="records")
    fragment = f'''<section id="lgbm-ndx-2016-viz">
  <h2>Current best LGBM vs Nasdaq-100, 2016-2026</h2>
  <div class="viz-row" id="ln16-legend" aria-label="Series"></div>
  <div id="ln16-chart"></div>
  <div class="tooltip" id="ln16-tooltip" role="tooltip" hidden></div>
  <p class="sr-only" id="ln16-summary">Normalized net value comparison of the current best 75/25 LightGBM A-share strategy and the Nasdaq-100 price index.</p>
</section>
<style>
  #lgbm-ndx-2016-viz {{ position: relative; width: 100%; color: var(--foreground); }}
  #lgbm-ndx-2016-viz h2 {{ margin: 0 0 8px; font-weight: 500; letter-spacing: 0; }}
  #lgbm-ndx-2016-viz #ln16-legend {{ margin-bottom: 8px; gap: 14px; }}
  #lgbm-ndx-2016-viz .series-toggle {{ background: transparent; border: 0; color: var(--foreground); padding: 3px 0; display: inline-flex; align-items: center; gap: 6px; }}
  #lgbm-ndx-2016-viz .series-toggle[aria-pressed="false"] {{ opacity: 0.48; }}
  #lgbm-ndx-2016-viz .swatch {{ width: 20px; height: 3px; display: inline-block; background: var(--swatch); }}
  #lgbm-ndx-2016-viz #ln16-chart {{ width: 100%; min-height: 370px; }}
  #lgbm-ndx-2016-viz .plot {{ display: block; overflow: visible; }}
  #lgbm-ndx-2016-viz .axis text, #lgbm-ndx-2016-viz .axis-title, #lgbm-ndx-2016-viz .end-label {{ fill: var(--foreground); font-size: 12px; }}
  #lgbm-ndx-2016-viz .axis path, #lgbm-ndx-2016-viz .axis line {{ stroke: var(--border); }}
  #lgbm-ndx-2016-viz .grid line {{ stroke: var(--border); stroke-opacity: 0.55; }}
  #lgbm-ndx-2016-viz .grid path {{ display: none; }}
  #lgbm-ndx-2016-viz [data-chart-frame] {{ fill: transparent; stroke: var(--border); }}
  #lgbm-ndx-2016-viz .tooltip {{ position: absolute; pointer-events: none; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); padding: 8px 10px; border-radius: 6px; font-size: 12px; z-index: 5; }}
  #lgbm-ndx-2016-viz .tooltip-row {{ display: flex; justify-content: space-between; gap: 18px; white-space: nowrap; }}
  #lgbm-ndx-2016-viz .tooltip-date {{ margin-bottom: 4px; font-weight: 500; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById("lgbm-ndx-2016-viz");
  const chart = document.getElementById("ln16-chart");
  const legend = document.getElementById("ln16-legend");
  const tooltip = document.getElementById("ln16-tooltip");
  const data = {json.dumps(data, ensure_ascii=True, separators=(',', ':'))};
  const series = [
    {{key:"current_best_lgbm",label:"Current best 75/25 LGBM",color:"var(--viz-series-1)"}},
    {{key:"nasdaq_100",label:"Nasdaq-100 (NDX)",color:"var(--viz-series-2)"}}
  ];
  const parseDate = d3.timeParse("%Y-%m-%d");
  data.forEach(d => d.x = parseDate(d.date));
  const visible = new Set(series.map(d => d.key));
  series.forEach(item => {{
    const button = document.createElement("button");
    button.type = "button";
    button.className = "series-toggle";
    button.setAttribute("aria-pressed", "true");
    button.innerHTML = `<span class="swatch" style="--swatch:${{item.color}}"></span><span>${{item.label}}</span>`;
    button.addEventListener("click", () => {{
      if (visible.has(item.key) && visible.size > 1) visible.delete(item.key); else visible.add(item.key);
      button.setAttribute("aria-pressed", String(visible.has(item.key)));
      draw();
    }});
    legend.appendChild(button);
  }});
  function draw() {{
    chart.replaceChildren();
    const width = Math.max(320, chart.getBoundingClientRect().width || 736);
    const height = width <= 400 ? 380 : 440;
    const margin = {{top:18,right:width <= 400 ? 16 : 180,bottom:54,left:68}};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const active = series.filter(s => visible.has(s.key));
    const extent = d3.extent(active.flatMap(s => data.map(d => d[s.key])));
    const padding = Math.max((extent[1] - extent[0]) * 0.06, 0.05);
    const x = d3.scaleTime().domain(d3.extent(data, d => d.x)).range([0, innerWidth]);
    const y = d3.scaleLinear().domain([Math.max(0, extent[0] - padding), extent[1] + padding]).nice().range([innerHeight, 0]);
    const svg = d3.select(chart).append("svg").attr("class","plot").attr("viewBox",`0 0 ${{width}} ${{height}}`).attr("width",width).attr("height",height).attr("role","img").attr("aria-labelledby","ln16-chart-title ln16-chart-desc");
    svg.append("title").attr("id","ln16-chart-title").text("Current best LGBM versus Nasdaq-100");
    svg.append("desc").attr("id","ln16-chart-desc").text(document.getElementById("ln16-summary").textContent);
    const g = svg.append("g").attr("transform",`translate(${{margin.left}},${{margin.top}})`);
    g.append("rect").attr("data-chart-frame","").attr("width",innerWidth).attr("height",innerHeight);
    g.append("g").attr("class","grid").call(d3.axisLeft(y).ticks(6).tickSize(-innerWidth).tickFormat(""));
    g.append("g").attr("class","axis").attr("transform",`translate(0,${{innerHeight}})`).call(d3.axisBottom(x).ticks(width <= 400 ? 4 : 7).tickFormat(d3.timeFormat("%Y")));
    g.append("g").attr("class","axis").call(d3.axisLeft(y).ticks(6).tickFormat(d => `${{d.toFixed(1)}}x`));
    g.append("text").attr("class","axis-title").attr("data-axis","x").attr("x",innerWidth/2).attr("y",innerHeight+44).attr("text-anchor","middle").text("Calendar date");
    g.append("text").attr("class","axis-title").attr("data-axis","y").attr("transform","rotate(-90)").attr("x",-innerHeight/2).attr("y",-52).attr("text-anchor","middle").text("Normalized net value (x)");
    const line = key => d3.line().x(d => x(d.x)).y(d => y(d[key]))(data);
    active.forEach(item => {{
      g.append("path").attr("data-series",item.key).attr("fill","none").attr("stroke",item.color).attr("stroke-width",2.4).attr("d",line(item.key));
      if (width > 400) {{ const end=data[data.length-1]; g.append("text").attr("class","end-label").attr("x",innerWidth+9).attr("y",y(end[item.key])+4).text(`${{item.label}} ${{end[item.key].toFixed(2)}}x`); }}
    }});
    const guide=g.append("line").attr("data-chart-hover-guide","").attr("y1",0).attr("y2",innerHeight).attr("stroke","var(--foreground)").attr("stroke-opacity",0.35).style("display","none");
    const markers=new Map(active.map(item=>[item.key,g.append("circle").attr("data-chart-hover-marker",item.key).attr("r",4).attr("fill",item.color).style("display","none")]));
    const bisect=d3.bisector(d=>d.x).center;
    g.append("rect").attr("data-chart-hit","").attr("data-chart-hover-overlay","cross-series").attr("width",innerWidth).attr("height",innerHeight).attr("fill","transparent")
      .on("pointermove",event=>{{ const [px]=d3.pointer(event); const target=x.invert(px); const index=bisect(data,target); const right=data[Math.max(0,Math.min(data.length-1,index))]; const left=data[Math.max(0,index-1)]; const span=Math.max(right.x-left.x,1); const ratio=Math.max(0,Math.min(1,(target-left.x)/span)); const interpolated=Object.fromEntries(active.map(item=>[item.key,left[item.key]+(right[item.key]-left[item.key])*ratio])); const gx=x(target); guide.attr("x1",gx).attr("x2",gx).style("display",null); active.forEach(item=>markers.get(item.key).attr("cx",gx).attr("cy",y(interpolated[item.key])).style("display",null)); tooltip.innerHTML=`<div class="tooltip-date">${{d3.timeFormat("%Y-%m-%d")(target)}}</div>`+active.map(item=>`<div class="tooltip-row"><span>${{item.label}}</span><strong>${{interpolated[item.key].toFixed(2)}}x</strong></div>`).join(""); tooltip.hidden=false; tooltip.style.left=`${{Math.max(0,Math.min(root.clientWidth-220,margin.left+gx+12))}}px`; tooltip.style.top=`${{Math.max(42,margin.top+8)}}px`; }})
      .on("pointerleave",()=>{{ guide.style("display","none"); markers.forEach(marker=>marker.style("display","none")); tooltip.hidden=true; }});
  }}
  draw();
  new ResizeObserver(draw).observe(chart);
}})();
</script>
'''
    OUTPUT_HTML.write_text(fragment, encoding="utf-8")


def write_report(payload: dict) -> None:
    strategy = payload["strategy"]["metrics"]
    ndx = payload["nasdaq_100"]["metrics"]
    lines = [
        "# Current best LGBM vs Nasdaq-100, 2016-2026",
        "",
        f"- Window: {payload['window']['common_start']} to {payload['window']['common_end']}",
        "- Both curves are normalized to 1.00 on the same first effective date.",
        "- Strategy: 40% rule protection + 60% model rank; model rank is 75% Alpha158-lite + 25% Alpha158+Barra.",
        "- Execution: T-close signal, T+1 open fill, Top5, 10-session rebalance, at most one replacement, 12 bps one-way cost.",
        "- Benchmark: Nasdaq-100 NDX price index; excludes dividends, fees, taxes and FX.",
        "",
        "| Curve | Terminal value | Total return | CAGR | Sharpe (0% rate) | Max drawdown |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Current best 75/25 LGBM | {strategy['terminal_value']:.2f}x | {strategy['total_return']:.2%} | {strategy['annual_return']:.2%} | {strategy['sharpe_zero_rate']:.3f} | {strategy['max_drawdown']:.2%} |",
        f"| Nasdaq-100 NDX | {ndx['terminal_value']:.2f}x | {ndx['total_return']:.2%} | {ndx['annual_return']:.2%} | {ndx['sharpe_zero_rate']:.3f} | {ndx['max_drawdown']:.2%} |",
        "",
        "## Calendar-year returns",
        "",
        "| Year | Current best LGBM | Nasdaq-100 |",
        "|---|---:|---:|",
    ]
    years = sorted(set(payload["strategy"]["yearly_returns"]) | set(payload["nasdaq_100"]["yearly_returns"]))
    for year in years:
        strategy_value = payload["strategy"]["yearly_returns"].get(year, float("nan"))
        ndx_value = payload["nasdaq_100"]["yearly_returns"].get(year, float("nan"))
        lines.append(f"| {year} | {strategy_value:.2%} | {ndx_value:.2%} |")
    lines.extend([
        "",
        "## Audit and limitations",
        "",
        f"- Every monthly refit passed the mature-label check: {payload['training_audit']['all_strictly_mature']}.",
        *[f"- {item}" for item in payload["limitations"]],
    ])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    source = json.loads(SOURCE_OPTIMIZATION.read_text(encoding="utf-8"))
    universe, snapshot_id = frozen_research_candidates(source["generated_at"])
    history, requested_codes, warnings = load_current_history(universe)
    history["date"] = pd.to_datetime(history["date"])
    history = history[history["date"].le(pd.Timestamp(END))].copy()
    frame, feature_sets = build_extended_feature_frame(history)
    frame = frame.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)

    print("training Alpha158-lite monthly expanding branch", flush=True)
    base_prediction, base_audit = fit_expanding_lgbm_predictions(
        frame, feature_sets["alpha158_baseline"], WINNER_CONFIG, START, END
    )
    print("training Alpha158+Barra monthly expanding branch", flush=True)
    barra_prediction, barra_audit = fit_expanding_lgbm_predictions(
        frame, feature_sets["alpha158_plus_barra"], WINNER_CONFIG, START, END
    )
    prediction = blend_model_predictions(
        frame, base_prediction, barra_prediction, BARRA_WEIGHT
    )
    rules = baseline_scores(frame, START, END)
    scores = blend_scores(
        frame, prediction, rules, WINNER_CONFIG["baseline_weight"]
    )
    scores["expected_return"] = 0.0
    scores["downside_probability"] = 0.0

    print("running common next-open portfolio replay", flush=True)
    result = multihead_portfolio_backtest(
        history,
        scores,
        PORTFOLIO_CANDIDATES["current_rank_top5"],
        START,
        END,
        rebalance_days=REBALANCE_DAYS,
        n_drop=N_DROP,
        cost_bps=COST_BPS,
    )
    strategy = normalized_curve(result, "strategy_return")
    strategy.index = pd.to_datetime(strategy.index)
    ndx, ndx_source = fetch_ndx()
    strategy, ndx, display = build_common_curves(strategy, ndx)

    strategy_metrics = curve_metrics(strategy)
    ndx_metrics = curve_metrics(ndx)
    strict_checks = {
        "alpha158_baseline": maturity_checks(base_audit),
        "alpha158_plus_barra": maturity_checks(barra_audit),
    }
    all_strict = bool(strict_checks) and all(
        checks and all(checks.values()) for checks in strict_checks.values()
    )
    if not all_strict:
        raise RuntimeError("at least one monthly model used an immature training label")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    display.to_csv(OUTPUT_CSV, encoding="utf-8-sig", float_format="%.10f")
    render_svg(display, strategy_metrics, ndx_metrics)
    render_inline_html(display)

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "revealed_historical_diagnostic_not_promoted",
        "window": {
            "requested_start": START.isoformat(),
            "requested_end": END.isoformat(),
            "common_start": display.index[0].date().isoformat(),
            "common_end": display.index[-1].date().isoformat(),
        },
        "strategy": {
            "name": "Current best 75/25 LGBM",
            "architecture": "40% rules + 60% model; model = 75% Alpha158-lite + 25% Alpha158+Barra",
            "model_active_from": min(base_audit["maturity_audit"], key=lambda row: row["prediction_start"])["prediction_start"],
            "pre_model_behavior": "Before sufficient mature training history, model predictions are unavailable and ranking falls back to the rules branch.",
            "config": WINNER_CONFIG,
            "feature_counts": {
                "alpha158_lite": len(feature_sets["alpha158_baseline"]),
                "alpha158_plus_barra": len(feature_sets["alpha158_plus_barra"]),
            },
            "execution": "T-close signal; T+1 open; Top5; 10-session rebalance; max one replacement; 12bp one-way cost",
            "metrics": strategy_metrics,
            "yearly_returns": yearly_returns(strategy),
            "backtest_metrics": result["metrics"],
        },
        "nasdaq_100": {
            "name": "Nasdaq-100",
            "symbol": "NDX",
            "return_type": "price return; excludes dividends, fees, taxes and FX",
            "source_status": ndx_source,
            "metrics": ndx_metrics,
            "yearly_returns": yearly_returns(ndx),
        },
        "training_audit": {
            "all_strictly_mature": all_strict,
            "refit_months": {
                "alpha158_lite": base_audit["refit_months"],
                "alpha158_plus_barra": barra_audit["refit_months"],
            },
            "checks": strict_checks,
        },
        "universe": {
            "snapshot_run_id": snapshot_id,
            "requested_codes": len(requested_codes),
            "loaded_codes": int(history["code"].nunique()),
            "warnings": warnings,
        },
        "artifacts": {
            "json": str(OUTPUT_JSON.relative_to(ROOT_DIR)),
            "csv": str(OUTPUT_CSV.relative_to(ROOT_DIR)),
            "report": str(OUTPUT_REPORT.relative_to(ROOT_DIR)),
            "svg": str(OUTPUT_SVG.relative_to(ROOT_DIR)),
            "inline_html": str(OUTPUT_HTML),
        },
        "limitations": [
            "The LGBM branch first becomes active in November 2017; earlier 2016-2017 ranking falls back to the rules branch because mature training history is insufficient.",
            "The current Top50 universe is backfilled and has survivorship and constituent bias.",
            "Current-vintage adjusted A-share prices are not strict point-in-time prices.",
            "This 2016-2026 window has been repeatedly revealed and is not a clean untouched holdout.",
            "NDX is a price index and excludes dividends; strategy return includes modeled trading costs.",
            "No USD/CNY conversion is applied; both curves are normalized local-currency returns.",
            "China and US holidays differ; calendar forward filling is used only for display.",
        ],
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(json.dumps({
        "window": payload["window"],
        "strategy": strategy_metrics,
        "nasdaq_100": ndx_metrics,
        "training_audit": payload["training_audit"],
        "ndx_source": ndx_source,
        "artifacts": payload["artifacts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
