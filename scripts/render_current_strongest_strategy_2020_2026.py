from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CURVES = ROOT / "reports" / "semiconductor_trend_csi300_hedge_v1_curves.csv"
INDEX_PATH = ROOT / "data" / "cache" / "index" / "csi300_ohlc_2017_2026.csv"
OUTPUT_PNG = ROOT / "reports" / "current_strongest_strategy_2020_2026.png"
OUTPUT_CSV = ROOT / "reports" / "current_strongest_strategy_2020_2026_curve.csv"
OUTPUT_JSON = ROOT / "data" / "current_strongest_strategy_2020_2026_chart.json"
OUTPUT_HTML = (
    Path(tempfile.gettempdir())
    / "codex-visualizations"
    / "current-strongest-strategy-2020-2026.html"
)

START = "2020-01-02"
END = "2026-08-25"
SERIES = {
    "strongest": "Trend expert + CSI300 hedge 50%",
    "trend_unhedged": "Trend expert, unhedged",
    "current_lgbm": "Current LGBM",
    "csi300": "CSI300 price index",
}
def metrics(curve: pd.Series) -> dict:
    returns = curve.pct_change().fillna(0.0)
    total_return = float(curve.iloc[-1] / curve.iloc[0] - 1.0)
    annual_return = float((1.0 + total_return) ** (252.0 / len(curve)) - 1.0)
    volatility = float(returns.std(ddof=0) * np.sqrt(252.0))
    drawdown = curve.div(curve.cummax()).sub(1.0)
    return {
        "terminal_value": float(curve.iloc[-1]),
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": float(returns.mean() * 252.0 / volatility) if volatility > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "observations": int(len(curve)),
    }


def load_data() -> pd.DataFrame:
    strategy = (
        pd.read_csv(SOURCE_CURVES, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .loc[START:END]
    )
    index = (
        pd.read_csv(INDEX_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()["close"]
        .reindex(strategy.index)
        .ffill()
    )
    output = pd.DataFrame(
        {
            "strongest": strategy["trend_hedge_50"],
            "trend_unhedged": strategy["trend_expert"],
            "current_lgbm": strategy["current_lgbm"],
            "csi300": index,
        }
    ).dropna()
    output = output.div(output.iloc[0])
    for column in SERIES:
        output[f"{column}_drawdown"] = output[column].div(
            output[column].cummax()
        ).sub(1.0)
    output.index.name = "date"
    return output


def html_fragment(records: list[dict], summaries: dict[str, dict]) -> str:
    template = r'''
<section id="strongest-strategy-2020-2026" style="position:relative; width:100%;">
  <h2>2020-2026 当前最强历史候选回测</h2>
  <div class="text-muted strongest-subtitle">半导体趋势专家 + CSI300 MA120 动态对冲（最高 50%） · 起点净值 1.00</div>
  <div class="viz-row strongest-metrics" aria-label="核心回测指标">
    <span>期末 <strong>__TERMINAL__x</strong></span>
    <span>年化 <strong>__CAGR__</strong></span>
    <span>Sharpe <strong>__SHARPE__</strong></span>
    <span>最大回撤 <strong>__MDD__</strong></span>
  </div>
  <div class="viz-row strongest-legend" aria-label="曲线显示选择"></div>
  <div class="strongest-panel">
    <div class="text-small text-muted strongest-panel-title">归一化净值（倍）</div>
    <svg class="strongest-main" role="img" aria-label="2020 至 2026 策略与 CSI300 归一化净值曲线"></svg>
  </div>
  <div class="strongest-panel">
    <div class="text-small text-muted strongest-panel-title">历史回撤（%）</div>
    <svg class="strongest-dd" role="img" aria-label="2020 至 2026 各策略回撤曲线"></svg>
  </div>
  <div class="text-small text-muted strongest-note">2020-01-02 至 2026-08-25 · T 日收盘信号，T+1 开盘执行 · CSI300 价格指数未含分红 · 历史诊断，非新样本外</div>
  <div class="tooltip strongest-tooltip" role="tooltip" hidden></div>
</section>
<style>
#strongest-strategy-2020-2026 { color: var(--foreground); }
#strongest-strategy-2020-2026 h2 { margin: 0 0 4px; font-weight: 500; letter-spacing: 0; }
#strongest-strategy-2020-2026 .strongest-subtitle { margin-bottom: 10px; }
#strongest-strategy-2020-2026 .strongest-metrics { gap: 18px; margin-bottom: 8px; }
#strongest-strategy-2020-2026 .strongest-metrics strong { font-weight: 500; color: var(--foreground); }
#strongest-strategy-2020-2026 .strongest-legend { gap: 12px; margin-bottom: 8px; align-items: center; }
#strongest-strategy-2020-2026 .strongest-legend button { background: transparent; border: 0; color: var(--foreground); padding: 3px 0; display: inline-flex; align-items: center; gap: 6px; }
#strongest-strategy-2020-2026 .strongest-legend button[aria-pressed="false"] { color: var(--muted-foreground); opacity: 0.55; }
#strongest-strategy-2020-2026 .strongest-swatch { width: 18px; height: 3px; display: inline-block; background: var(--series-color); }
#strongest-strategy-2020-2026 .strongest-panel { width: 100%; margin-top: 4px; }
#strongest-strategy-2020-2026 .strongest-panel-title { margin-left: 68px; }
#strongest-strategy-2020-2026 svg { width: 100%; display: block; overflow: visible; }
#strongest-strategy-2020-2026 .domain, #strongest-strategy-2020-2026 .tick line { stroke: var(--border); }
#strongest-strategy-2020-2026 .tick text { fill: var(--muted-foreground); font-size: 12px; }
#strongest-strategy-2020-2026 .axis-title { fill: var(--foreground); font-size: 12px; }
#strongest-strategy-2020-2026 .strongest-note { margin: 6px 0 0 68px; }
#strongest-strategy-2020-2026 .strongest-tooltip { position: absolute; pointer-events: none; z-index: 5; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); padding: 8px 10px; max-width: 260px; }
#strongest-strategy-2020-2026 .strongest-tooltip div { white-space: nowrap; }
@media (max-width: 480px) {
  #strongest-strategy-2020-2026 .strongest-metrics { gap: 8px 14px; }
  #strongest-strategy-2020-2026 .strongest-panel-title, #strongest-strategy-2020-2026 .strongest-note { margin-left: 62px; }
}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {
  const root = document.getElementById("strongest-strategy-2020-2026");
  const raw = __DATA__;
  const parseDate = d3.utcParse("%Y-%m-%d");
  const data = raw.map(d => ({...d, date: parseDate(d.date)}));
  const defs = [
    {key:"strongest", label:"趋势专家 + CSI300 50% 对冲", color:"var(--viz-series-1)", width:2.8},
    {key:"trend_unhedged", label:"趋势专家（未对冲）", color:"var(--viz-series-2)", width:1.8},
    {key:"current_lgbm", label:"当前正式 LGBM", color:"var(--viz-series-3)", width:1.8},
    {key:"csi300", label:"CSI300 价格指数", color:"var(--viz-series-4)", width:1.5}
  ];
  const visible = new Set(defs.map(d => d.key));
  const legend = d3.select(root).select(".strongest-legend");
  const buttons = legend.selectAll("button").data(defs).join("button")
    .attr("type", "button").attr("aria-pressed", "true")
    .on("click", function(event, d) {
      if (visible.has(d.key) && visible.size > 1) visible.delete(d.key); else visible.add(d.key);
      d3.select(this).attr("aria-pressed", visible.has(d.key) ? "true" : "false");
      drawAll();
    });
  buttons.append("span").attr("class", "strongest-swatch").style("--series-color", d => d.color);
  buttons.append("span").text(d => d.label);
  const tooltip = d3.select(root).select(".strongest-tooltip");
  const bisect = d3.bisector(d => d.date).left;

  function valueAt(key, target, isDrawdown) {
    const i = Math.max(1, Math.min(data.length - 1, bisect(data, target)));
    const a = data[i - 1], b = data[i];
    const ratio = (target - a.date) / Math.max(1, b.date - a.date);
    const field = isDrawdown ? `${key}_drawdown` : key;
    return a[field] + (b[field] - a[field]) * ratio;
  }

  function draw(selector, isDrawdown) {
    const svg = d3.select(root).select(selector);
    svg.selectAll("*").remove();
    const width = Math.max(320, svg.node().parentElement.getBoundingClientRect().width);
    const height = isDrawdown ? (width < 480 ? 220 : 245) : (width < 480 ? 310 : 390);
    const margin = {top: 10, right: 18, bottom: 48, left: width < 480 ? 62 : 68};
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("height", height);
    const selected = defs.filter(d => visible.has(d.key));
    const x = d3.scaleUtc().domain(d3.extent(data, d => d.date)).range([margin.left, width - margin.right]);
    const fields = selected.map(d => isDrawdown ? `${d.key}_drawdown` : d.key);
    const values = fields.flatMap(field => data.map(d => d[field]));
    const extent = d3.extent(values);
    const yDomain = isDrawdown
      ? [Math.min(-0.01, extent[0] * 1.08), 0.015]
      : [Math.min(0.95, extent[0] * 0.98), extent[1] * 1.04];
    const y = d3.scaleLinear().domain(yDomain).nice().range([height - margin.bottom, margin.top]);
    svg.append("rect").attr("data-chart-frame", "").attr("x", margin.left).attr("y", margin.top)
      .attr("width", width - margin.left - margin.right).attr("height", height - margin.top - margin.bottom)
      .attr("fill", "none").attr("stroke", "var(--border)");
    svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(width < 480 ? 4 : 7).tickSizeOuter(0));
    svg.append("g").attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(5).tickFormat(isDrawdown ? d3.format(".0%") : d => `${d.toFixed(1)}x`).tickSizeOuter(0));
    const line = field => d3.line().defined(d => Number.isFinite(d[field])).x(d => x(d.date)).y(d => y(d[field]));
    selected.forEach(def => {
      const field = isDrawdown ? `${def.key}_drawdown` : def.key;
      svg.append("path").datum(data).attr("fill", "none").attr("stroke", def.color)
        .attr("stroke-width", def.width).attr("stroke-dasharray", def.key === "csi300" ? "6 4" : null)
        .attr("d", line(field)).attr("data-series", def.key);
    });
    const guide = svg.append("line").attr("data-chart-hover-guide", "").attr("y1", margin.top)
      .attr("y2", height - margin.bottom).attr("stroke", "var(--border)").attr("pointer-events", "none").style("display", "none");
    const markers = svg.append("g").attr("pointer-events", "none");
    const overlay = svg.append("rect").attr("data-chart-hit", "").attr("data-chart-hover-overlay", "cross-series")
      .attr("x", margin.left).attr("y", margin.top).attr("width", width - margin.left - margin.right)
      .attr("height", height - margin.top - margin.bottom).attr("fill", "transparent")
      .on("pointermove", event => {
        const [px] = d3.pointer(event, svg.node());
        const clamped = Math.max(margin.left, Math.min(width - margin.right, px));
        const target = x.invert(clamped);
        guide.attr("x1", clamped).attr("x2", clamped).style("display", null);
        const rows = selected.map(def => ({def, value: valueAt(def.key, target, isDrawdown)}));
        markers.selectAll("circle").data(rows, d => d.def.key).join("circle")
          .attr("data-chart-hover-marker", "").attr("cx", clamped).attr("cy", d => y(d.value))
          .attr("r", 4).attr("fill", d => d.def.color).attr("stroke", "var(--background)").attr("stroke-width", 1.5);
        const dateText = d3.utcFormat("%Y-%m-%d")(target);
        tooltip.html(`<div><strong>${dateText}</strong></div>` + rows.map(r => `<div>${r.def.label}: ${isDrawdown ? d3.format(".2%")(r.value) : r.value.toFixed(2) + "x"}</div>`).join(""));
        tooltip.attr("hidden", null).style("left", `${Math.min(root.clientWidth - 270, clamped + 12)}px`).style("top", `${event.clientY - root.getBoundingClientRect().top + 12}px`);
      })
      .on("pointerleave", () => { guide.style("display", "none"); markers.selectAll("circle").remove(); tooltip.attr("hidden", true); });
    svg.append("text").attr("class", "axis-title").attr("data-axis", "x")
      .attr("x", (margin.left + width - margin.right) / 2).attr("y", height - 6).attr("text-anchor", "middle").text("交易日期");
    svg.append("text").attr("class", "axis-title").attr("data-axis", "y")
      .attr("transform", `translate(15,${(margin.top + height - margin.bottom) / 2}) rotate(-90)`)
      .attr("text-anchor", "middle").text(isDrawdown ? "回撤（%）" : "归一化净值（倍）");
  }
  function drawAll() { draw(".strongest-main", false); draw(".strongest-dd", true); }
  drawAll();
  new ResizeObserver(drawAll).observe(root);
})();
</script>
'''
    best = summaries["strongest"]
    return (
        template.replace("__DATA__", json.dumps(records, ensure_ascii=False, separators=(",", ":")))
        .replace("__TERMINAL__", f"{best['terminal_value']:.2f}")
        .replace("__CAGR__", f"{best['annual_return']:.2%}")
        .replace("__SHARPE__", f"{best['sharpe']:.3f}")
        .replace("__MDD__", f"{best['max_drawdown']:.2%}")
    )


def main() -> None:
    frame = load_data()
    summaries = {key: metrics(frame[key]) for key in SERIES}
    frame.to_csv(OUTPUT_CSV, encoding="utf-8-sig", float_format="%.10f")

    sampled = frame.iloc[::5].copy()
    if sampled.index[-1] != frame.index[-1]:
        sampled = pd.concat([sampled, frame.iloc[[-1]]])
    records = []
    for timestamp, row in sampled.iterrows():
        record = {"date": timestamp.date().isoformat()}
        for key in SERIES:
            record[key] = round(float(row[key]), 6)
            record[f"{key}_drawdown"] = round(float(row[f"{key}_drawdown"]), 6)
        records.append(record)
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html_fragment(records, summaries), encoding="utf-8")

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "historical_diagnostic_only_not_deployed",
        "window": [START, END],
        "strongest_historical_candidate": (
            "semiconductor trend expert plus CSI300 MA120 dynamic hedge, maximum 50%"
        ),
        "series": SERIES,
        "metrics": summaries,
        "assumptions": {
            "stock_execution": "T close signal; T+1 open execution; 12bp one-way cost",
            "hedge_execution": "CSI300 close versus MA120; next-open execution; 2bp change cost",
            "csi300": "price index, dividends excluded",
            "sample_status": "repeatedly revealed historical diagnostic, not fresh out-of-sample",
        },
        "artifacts": {
            "png": str(OUTPUT_PNG.relative_to(ROOT)),
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "html": str(OUTPUT_HTML),
        },
        "production_change": False,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "png": str(OUTPUT_PNG),
        "csv": str(OUTPUT_CSV),
        "json": str(OUTPUT_JSON),
        "html": str(OUTPUT_HTML),
        "metrics": summaries,
        "png_bytes": OUTPUT_PNG.stat().st_size if OUTPUT_PNG.exists() else None,
        "html_bytes": OUTPUT_HTML.stat().st_size,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
