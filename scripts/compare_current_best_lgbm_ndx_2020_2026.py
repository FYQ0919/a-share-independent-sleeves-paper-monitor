from __future__ import annotations

from datetime import datetime
import html
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp("2020-01-02")
END = pd.Timestamp("2026-08-25")
STRATEGY_SOURCE = ROOT / "reports" / "overnight_factor_lgbm_2020_2026_curves.csv"
STRATEGY_AUDIT_SOURCE = ROOT / "data" / "overnight_factor_lgbm_2020_2026.json"
NDX_FALLBACK = ROOT / "reports" / "best_lgbm_vs_nasdaq100_10y.csv"
OUTPUT_JSON = ROOT / "data" / "current_best_lgbm_vs_ndx_2020_2026.json"
OUTPUT_CSV = ROOT / "reports" / "current_best_lgbm_vs_ndx_2020_2026.csv"
OUTPUT_REPORT = ROOT / "reports" / "current_best_lgbm_vs_ndx_2020_2026.md"
OUTPUT_SVG = ROOT / "reports" / "current_best_lgbm_vs_ndx_2020_2026.svg"
OUTPUT_HTML = ROOT / "reports" / "current-best-lgbm-vs-ndx-2020-2026.html"
NDX_URL = (
    "https://api.nasdaq.com/api/quote/NDX/historical"
    "?assetclass=index&fromdate=2019-12-30&todate=2026-08-25&limit=4000"
)


def curve_metrics(values: pd.Series) -> dict[str, float | str | int]:
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        raise ValueError("curve requires at least two observations")
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
        "positive_day_rate": float(returns.gt(0).mean()),
    }


def yearly_returns(values: pd.Series) -> dict[str, float]:
    clean = values.dropna().astype(float)
    output = {}
    for year in sorted(clean.index.year.unique()):
        year_end = clean.loc[clean.index < pd.Timestamp(int(year) + 1, 1, 1)]
        before_year = clean.loc[clean.index < pd.Timestamp(int(year), 1, 1)]
        if year_end.empty:
            continue
        base = float(before_year.iloc[-1]) if not before_year.empty else float(clean.iloc[0])
        output[str(year)] = float(year_end.iloc[-1] / base - 1.0)
    return output


def load_strategy() -> tuple[pd.Series, dict]:
    audit = json.loads(STRATEGY_AUDIT_SOURCE.read_text(encoding="utf-8"))
    if audit.get("selected_for_forward_research") != "previous_lgbm":
        raise RuntimeError("latest research no longer selects previous_lgbm")
    frame = pd.read_csv(STRATEGY_SOURCE, parse_dates=["date"])
    values = frame.set_index("date")["previous_lgbm"].sort_index()
    values = values.loc[(values.index >= START) & (values.index <= END)]
    if values.empty or not np.isclose(float(values.iloc[0]), 1.0):
        raise RuntimeError("latest best LGBM curve is missing a normalized start")
    return values, audit


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
        values = frame.dropna(subset=["date", "ndx"]).drop_duplicates("date").set_index("date")["ndx"].sort_index()
        status = {
            "mode": "live_nasdaq_api",
            "source": NDX_URL,
            "fetched_at": datetime.now().astimezone().isoformat(),
        }
    except Exception as exc:
        frame = pd.read_csv(NDX_FALLBACK, parse_dates=["date"])
        values = frame.set_index("date")["ndx"].sort_index()
        values = values.loc[~values.index.weekday.isin([5, 6])]
        status = {
            "mode": "cached_fallback",
            "source": str(NDX_FALLBACK.relative_to(ROOT)),
            "fallback_reason": str(exc),
            "cache_modified_at": datetime.fromtimestamp(
                NDX_FALLBACK.stat().st_mtime
            ).astimezone().isoformat(),
        }
    values = values.loc[(values.index >= START) & (values.index <= END)].astype(float)
    if values.empty:
        raise RuntimeError("NDX has no observations in the comparison window")
    values = values.div(float(values.iloc[0]))
    return values, status


def build_display_curve(strategy: pd.Series, ndx: pd.Series) -> pd.DataFrame:
    common_start = max(strategy.index.min(), ndx.index.min())
    common_end = min(strategy.index.max(), ndx.index.max())
    calendar = pd.date_range(common_start, common_end, freq="D")
    display = pd.concat(
        {
            "current_best_lgbm": strategy.reindex(calendar).ffill(),
            "nasdaq_100": ndx.reindex(calendar).ffill(),
        },
        axis=1,
    ).dropna()
    display = display.div(display.iloc[0])
    if not np.allclose(display.iloc[0].to_numpy(dtype=float), 1.0):
        raise RuntimeError("comparison curves do not share a normalized start")
    display.index.name = "date"
    return display


def render_svg(curve: pd.DataFrame) -> None:
    columns = ["current_best_lgbm", "nasdaq_100"]
    labels = {
        "current_best_lgbm": "Current best 75/25 LGBM",
        "nasdaq_100": "Nasdaq-100 (NDX)",
    }
    colors = {"current_best_lgbm": "#176B87", "nasdaq_100": "#C2413B"}
    width, height = 1080, 610
    left, right, top, bottom = 82, 240, 74, 64
    plot_width, plot_height = width - left - right, height - top - bottom
    values = curve[columns].to_numpy(dtype=float)
    low = min(0.9, float(np.nanmin(values)))
    high = max(1.1, float(np.nanmax(values)))
    padding = max((high - low) * 0.06, 0.05)
    low, high = max(0.0, low - padding), high + padding

    def x(index: int) -> float:
        return left + plot_width * index / max(len(curve) - 1, 1)

    def y(value: float) -> float:
        return top + plot_height * (high - value) / max(high - low, 1e-12)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAFAF7"/>',
        '<text x="82" y="30" font-family="Segoe UI, Arial" font-size="20" font-weight="600" fill="#1F2933">Current best LGBM vs Nasdaq-100, 2020-2026</text>',
        '<text x="82" y="51" font-family="Segoe UI, Arial" font-size="11" fill="#52606D">Both normalized to 1.00 on 2020-01-02; LGBM after 12 bps one-way cost; NDX price return</text>',
    ]
    for tick in np.linspace(low, high, 6):
        yy = y(float(tick))
        parts.append(
            f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_width}" y2="{yy:.2f}" stroke="#D9E2E8" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 9}" y="{yy + 4:.2f}" text-anchor="end" font-family="Segoe UI, Arial" font-size="10" fill="#52606D">{tick:.2f}x</text>'
        )
    dates = pd.DatetimeIndex(curve.index)
    for year in range(dates.min().year, dates.max().year + 1):
        index = int(np.argmin(np.abs((dates - pd.Timestamp(year, 1, 1)).days)))
        xx = x(index)
        parts.append(
            f'<text x="{xx:.2f}" y="{height - 36}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="10" fill="#52606D">{year}</text>'
        )
    for column in columns:
        points = " ".join(
            f"{x(index):.2f},{y(float(value)):.2f}"
            for index, value in enumerate(curve[column])
        )
        parts.append(
            f'<polyline data-series="{column}" points="{points}" fill="none" stroke="{colors[column]}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        value = float(curve[column].iloc[-1])
        yy = y(value)
        parts.append(
            f'<circle cx="{left + plot_width}" cy="{yy:.2f}" r="3.5" fill="{colors[column]}"/>'
        )
        parts.append(
            f'<text data-end-label="{column}" x="{left + plot_width + 13}" y="{yy + 4:.2f}" font-family="Segoe UI, Arial" font-size="11" fill="#1F2933">{html.escape(labels[column])} {value:.2f}x</text>'
        )
    parts.extend([
        f'<text x="{left + plot_width / 2:.2f}" y="{height - 13}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="11" fill="#52606D">Calendar date</text>',
        f'<text x="18" y="{top + plot_height / 2:.2f}" transform="rotate(-90 18 {top + plot_height / 2:.2f})" text-anchor="middle" font-family="Segoe UI, Arial" font-size="11" fill="#52606D">Normalized net value (x)</text>',
        "</svg>",
    ])
    OUTPUT_SVG.write_text("\n".join(parts), encoding="utf-8")


def render_inline_html(curve: pd.DataFrame) -> None:
    sample_index = list(range(0, len(curve), 5))
    if sample_index[-1] != len(curve) - 1:
        sample_index.append(len(curve) - 1)
    sample = curve.iloc[sample_index].reset_index()
    sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
    data = sample.round(6).to_dict(orient="records")
    fragment = f'''<section id="current-best-lgbm-ndx-viz">
  <h2>Current best LGBM vs Nasdaq-100, 2020-2026</h2>
  <div class="viz-row" id="cbn-legend" aria-label="Series"></div>
  <div id="cbn-chart"></div>
  <div class="tooltip" id="cbn-tooltip" role="tooltip" hidden></div>
  <p class="sr-only" id="cbn-summary">Normalized net value comparison between the current best 75/25 LightGBM strategy and the Nasdaq-100 price index.</p>
</section>
<style>
  #current-best-lgbm-ndx-viz {{ width: 100%; color: var(--foreground); }}
  #current-best-lgbm-ndx-viz h2 {{ margin: 0 0 8px; font-weight: 500; letter-spacing: 0; }}
  #current-best-lgbm-ndx-viz #cbn-legend {{ margin-bottom: 8px; gap: 14px; }}
  #current-best-lgbm-ndx-viz .series-toggle {{ background: transparent; border: 0; color: var(--foreground); padding: 3px 0; display: inline-flex; align-items: center; gap: 6px; }}
  #current-best-lgbm-ndx-viz .series-toggle[aria-pressed="false"] {{ opacity: 0.48; }}
  #current-best-lgbm-ndx-viz .swatch {{ width: 20px; height: 3px; display: inline-block; background: var(--swatch); }}
  #current-best-lgbm-ndx-viz #cbn-chart {{ width: 100%; min-height: 360px; }}
  #current-best-lgbm-ndx-viz .plot {{ display: block; overflow: visible; }}
  #current-best-lgbm-ndx-viz .axis text, #current-best-lgbm-ndx-viz .axis-title, #current-best-lgbm-ndx-viz .end-label {{ fill: var(--foreground); font-size: 12px; }}
  #current-best-lgbm-ndx-viz .axis path, #current-best-lgbm-ndx-viz .axis line {{ stroke: var(--border); }}
  #current-best-lgbm-ndx-viz .grid line {{ stroke: var(--border); stroke-opacity: 0.55; }}
  #current-best-lgbm-ndx-viz .grid path {{ display: none; }}
  #current-best-lgbm-ndx-viz [data-chart-frame] {{ fill: transparent; stroke: var(--border); }}
  #current-best-lgbm-ndx-viz .tooltip {{ position: absolute; pointer-events: none; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); padding: 8px 10px; border-radius: 6px; font-size: 12px; z-index: 5; }}
  #current-best-lgbm-ndx-viz .tooltip-row {{ display: flex; justify-content: space-between; gap: 18px; white-space: nowrap; }}
  #current-best-lgbm-ndx-viz .tooltip-date {{ margin-bottom: 4px; font-weight: 500; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById("current-best-lgbm-ndx-viz");
  const chart = document.getElementById("cbn-chart");
  const legend = document.getElementById("cbn-legend");
  const tooltip = document.getElementById("cbn-tooltip");
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
    const height = width <= 400 ? 370 : 430;
    const margin = {{top:18,right:width <= 400 ? 16 : 175,bottom:54,left:68}};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const active = series.filter(s => visible.has(s.key));
    const extent = d3.extent(active.flatMap(s => data.map(d => d[s.key])));
    const padding = Math.max((extent[1] - extent[0]) * 0.06, 0.05);
    const x = d3.scaleTime().domain(d3.extent(data, d => d.x)).range([0, innerWidth]);
    const y = d3.scaleLinear().domain([Math.max(0, Math.min(0.9, extent[0] - padding)), extent[1] + padding]).nice().range([innerHeight, 0]);
    const svg = d3.select(chart).append("svg").attr("class","plot").attr("viewBox",`0 0 ${{width}} ${{height}}`).attr("width",width).attr("height",height).attr("role","img").attr("aria-labelledby","cbn-chart-title cbn-chart-desc");
    svg.append("title").attr("id","cbn-chart-title").text("Current best LGBM versus Nasdaq-100");
    svg.append("desc").attr("id","cbn-chart-desc").text(document.getElementById("cbn-summary").textContent);
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
      .on("pointermove",event=>{{ const [px]=d3.pointer(event); const target=x.invert(px); const index=bisect(data,target); const right=data[Math.max(0,Math.min(data.length-1,index))]; const left=data[Math.max(0,index-1)]; const span=Math.max(right.x-left.x,1); const ratio=Math.max(0,Math.min(1,(target-left.x)/span)); const interpolated=Object.fromEntries(active.map(item=>[item.key,left[item.key]+(right[item.key]-left[item.key])*ratio])); const gx=x(target); guide.attr("x1",gx).attr("x2",gx).style("display",null); active.forEach(item=>markers.get(item.key).attr("cx",gx).attr("cy",y(interpolated[item.key])).style("display",null)); tooltip.innerHTML=`<div class="tooltip-date">${{d3.timeFormat("%Y-%m-%d")(target)}}</div>`+active.map(item=>`<div class="tooltip-row"><span>${{item.label}}</span><strong>${{interpolated[item.key].toFixed(2)}}x</strong></div>`).join(""); tooltip.hidden=false; tooltip.style.left=`${{Math.min(root.clientWidth-210,margin.left+gx+12)}}px`; tooltip.style.top=`${{Math.max(42,margin.top+8)}}px`; }})
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
        "# Current best LGBM vs Nasdaq-100, 2020-2026",
        "",
        f"- Window: {payload['window']['common_start']} to {payload['window']['common_end']}",
        "- Both curves are normalized to 1.00 on the common start date.",
        "- Strategy: 40% rule protection + 60% model rank; model rank is 75% Alpha158-lite + 25% Alpha158+Barra.",
        "- Execution: T-close signal, T+1 open fill, Top5, 10-session rebalance, at most one replacement, 12 bps one-way cost.",
        "- Benchmark: Nasdaq-100 NDX price index; excludes dividends, fees, taxes and FX.",
        "",
        "| Curve | Terminal value | Total return | CAGR | Sharpe (0% rate) | Max drawdown |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Current best 75/25 LGBM | {strategy['terminal_value']:.2f}x | {strategy['total_return']:.2%} | {strategy['annual_return']:.2%} | {strategy['sharpe_zero_rate']:.3f} | {strategy['max_drawdown']:.2%} |",
        f"| Nasdaq-100 NDX | {ndx['terminal_value']:.2f}x | {ndx['total_return']:.2%} | {ndx['annual_return']:.2%} | {ndx['sharpe_zero_rate']:.3f} | {ndx['max_drawdown']:.2%} |",
        "",
        "## Calendar-year price returns",
        "",
        "| Year | Current best LGBM | Nasdaq-100 |",
        "|---|---:|---:|",
    ]
    for year in sorted(payload["strategy"]["yearly_returns"]):
        lines.append(
            f"| {year} | {payload['strategy']['yearly_returns'][year]:.2%} | "
            f"{payload['nasdaq_100']['yearly_returns'].get(year, float('nan')):.2%} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        payload["decision"],
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in payload["limitations"]],
    ])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    strategy, audit = load_strategy()
    ndx, ndx_source = fetch_ndx()
    common_start = max(strategy.index.min(), ndx.index.min())
    common_end = min(strategy.index.max(), ndx.index.max())
    strategy_native = strategy.loc[(strategy.index >= common_start) & (strategy.index <= common_end)]
    strategy_native = strategy_native.div(float(strategy_native.iloc[0]))
    ndx_native = ndx.loc[(ndx.index >= common_start) & (ndx.index <= common_end)]
    ndx_native = ndx_native.div(float(ndx_native.iloc[0]))
    display = build_display_curve(strategy_native, ndx_native)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    display.to_csv(OUTPUT_CSV, encoding="utf-8-sig", float_format="%.10f")
    render_svg(display)
    render_inline_html(display)
    strategy_metrics = curve_metrics(strategy_native)
    ndx_metrics = curve_metrics(ndx_native)
    annual_lead = strategy_metrics["annual_return"] - ndx_metrics["annual_return"]
    drawdown_difference = strategy_metrics["max_drawdown"] - ndx_metrics["max_drawdown"]
    decision = (
        f"The current best LGBM outperformed Nasdaq-100 by {annual_lead:.2%} annualized over this revealed historical window. "
        f"Its maximum drawdown was {abs(drawdown_difference):.2%} {'smaller' if drawdown_difference > 0 else 'larger'} than NDX. "
        "This is a historical diagnostic and does not establish deployable forward alpha."
    )
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "research_only_not_promoted",
        "window": {
            "requested_start": START.date().isoformat(),
            "requested_end": END.date().isoformat(),
            "common_start": common_start.date().isoformat(),
            "common_end": common_end.date().isoformat(),
        },
        "strategy": {
            "name": "Current best 75/25 LGBM",
            "architecture": audit["strategies"]["previous_lgbm"],
            "execution": audit["execution"],
            "monthly_refits": audit["training_audit"]["refit_months"],
            "all_training_labels_strictly_mature": audit["training_audit"]["all_strictly_mature"],
            "metrics": strategy_metrics,
            "yearly_returns": yearly_returns(strategy_native),
        },
        "nasdaq_100": {
            "name": "Nasdaq-100",
            "symbol": "NDX",
            "return_type": "price return; excludes dividends, fees, taxes and FX",
            "data_status": ndx_source,
            "metrics": ndx_metrics,
            "yearly_returns": yearly_returns(ndx_native),
        },
        "comparison": {
            "annual_return_lead": annual_lead,
            "terminal_value_difference": strategy_metrics["terminal_value"] - ndx_metrics["terminal_value"],
            "max_drawdown_difference": drawdown_difference,
        },
        "decision": decision,
        "production_change": False,
        "artifacts": {
            "curve_csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "curve_svg": str(OUTPUT_SVG.relative_to(ROOT)),
            "interactive_html": str(OUTPUT_HTML.relative_to(ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(ROOT)),
        },
        "limitations": [
            "The current Top50 universe is backfilled and has survivorship and constituent bias.",
            "Current-vintage adjusted A-share prices are not strict point-in-time prices.",
            "NDX is a price index and excludes dividends; strategy returns include modeled trading costs.",
            "No USD/CNY FX conversion is applied because both curves are normalized local-currency returns.",
            "China and US holidays differ; calendar forward filling is used only for curve display.",
            "The 2020-2026 window has been repeatedly revealed and is not a clean promotion holdout.",
        ],
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(payload)
    print(json.dumps({
        "status": payload["status"],
        "ndx_data_status": ndx_source,
        "strategy_metrics": strategy_metrics,
        "nasdaq_100_metrics": ndx_metrics,
        "comparison": payload["comparison"],
        "artifacts": payload["artifacts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
