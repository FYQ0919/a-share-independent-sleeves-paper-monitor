from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "reports" / "current_strongest_strategy_2020_2026_curve.csv"
OUTPUT_CSV = ROOT / "reports" / "equal_weight_strategy_blend_2020_2026.csv"
OUTPUT_JSON = ROOT / "data" / "equal_weight_strategy_blend_2020_2026.json"
OUTPUT_PNG = ROOT / "reports" / "equal_weight_strategy_blend_2020_2026.png"
OUTPUT_HTML = (
    Path(tempfile.gettempdir())
    / "codex-visualizations"
    / "equal-weight-strategy-blend-2020-2026.html"
)

START = "2020-01-02"
END = "2026-08-25"
SERIES = {
    "equal_weight_blend": "50/50 equal-capital blend",
    "strongest_candidate": "Trend expert + CSI300 hedge 50%",
    "formal_lgbm": "Previous formal LGBM",
}


def performance(curve: pd.Series) -> dict:
    clean = pd.to_numeric(curve, errors="coerce").dropna()
    returns = clean.pct_change().fillna(0.0)
    total_return = float(clean.iloc[-1] / clean.iloc[0] - 1.0)
    annual_return = float((1.0 + total_return) ** (252.0 / len(clean)) - 1.0)
    volatility = float(returns.std(ddof=0) * np.sqrt(252.0))
    return {
        "terminal_value": float(clean.iloc[-1]),
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": float(returns.mean() * 252.0 / volatility) if volatility > 0 else 0.0,
        "max_drawdown": float(clean.div(clean.cummax()).sub(1.0).min()),
        "observations": int(len(clean)),
    }


def build_frame() -> pd.DataFrame:
    source = (
        pd.read_csv(INPUT_PATH, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .loc[START:END]
    )
    output = pd.DataFrame(
        {
            "strongest_candidate": source["strongest"],
            "formal_lgbm": source["current_lgbm"],
        }
    ).dropna()
    output = output.div(output.iloc[0])
    # Initial capital is split equally. Each sleeve then compounds independently;
    # there is no additional forced daily rebalance between the two sleeves.
    output["equal_weight_blend"] = (
        output["strongest_candidate"].mul(0.50)
        + output["formal_lgbm"].mul(0.50)
    )
    output = output[["equal_weight_blend", "strongest_candidate", "formal_lgbm"]]
    for column in SERIES:
        output[f"{column}_drawdown"] = output[column].div(
            output[column].cummax()
        ).sub(1.0)
    output.index.name = "date"
    return output


def html_fragment(records: list[dict], summaries: dict[str, dict]) -> str:
    template = r'''
<section id="equal-weight-strategy-blend" style="position:relative; width:100%;">
  <h2>2020-2026 两策略 50/50 等资金组合</h2>
  <div class="text-muted blend-subtitle">初始资金各 50%，子策略独立复利，不额外强制日间再平衡</div>
  <div class="viz-row blend-metrics" aria-label="组合回测指标">
    <span>期末 <strong>__TERMINAL__x</strong></span>
    <span>年化 <strong>__CAGR__</strong></span>
    <span>Sharpe <strong>__SHARPE__</strong></span>
    <span>最大回撤 <strong>__MDD__</strong></span>
  </div>
  <div class="viz-row blend-legend" aria-label="曲线显示选择"></div>
  <div class="blend-panel">
    <div class="text-small text-muted blend-panel-title">归一化净值（倍）</div>
    <svg class="blend-main" role="img" aria-label="2020 至 2026 两策略及等资金组合净值曲线"></svg>
  </div>
  <div class="blend-panel">
    <div class="text-small text-muted blend-panel-title">历史回撤（%）</div>
    <svg class="blend-dd" role="img" aria-label="2020 至 2026 两策略及组合回撤曲线"></svg>
  </div>
  <div class="text-small text-muted blend-note">2020-01-02 至 2026-08-25 · 子策略净收益已包含原有成本 · 未计额外跨账户调拨成本 · 历史诊断，非新样本外</div>
  <div class="tooltip blend-tooltip" role="tooltip" hidden></div>
</section>
<style>
#equal-weight-strategy-blend { color: var(--foreground); }
#equal-weight-strategy-blend h2 { margin: 0 0 4px; font-weight: 500; letter-spacing: 0; }
#equal-weight-strategy-blend .blend-subtitle { margin-bottom: 10px; }
#equal-weight-strategy-blend .blend-metrics { gap: 18px; margin-bottom: 8px; }
#equal-weight-strategy-blend .blend-metrics strong { font-weight: 500; color: var(--foreground); }
#equal-weight-strategy-blend .blend-legend { gap: 12px; margin-bottom: 8px; align-items: center; }
#equal-weight-strategy-blend .blend-legend button { background: transparent; border: 0; color: var(--foreground); padding: 3px 0; display: inline-flex; align-items: center; gap: 6px; }
#equal-weight-strategy-blend .blend-legend button[aria-pressed="false"] { color: var(--muted-foreground); opacity: 0.55; }
#equal-weight-strategy-blend .blend-swatch { width: 18px; height: 3px; display: inline-block; background: var(--series-color); }
#equal-weight-strategy-blend .blend-panel { width: 100%; margin-top: 4px; }
#equal-weight-strategy-blend .blend-panel-title { margin-left: 68px; }
#equal-weight-strategy-blend svg { width: 100%; display: block; overflow: visible; }
#equal-weight-strategy-blend .domain, #equal-weight-strategy-blend .tick line { stroke: var(--border); }
#equal-weight-strategy-blend .tick text { fill: var(--muted-foreground); font-size: 12px; }
#equal-weight-strategy-blend .axis-title { fill: var(--foreground); font-size: 12px; }
#equal-weight-strategy-blend .blend-note { margin: 6px 0 0 68px; }
#equal-weight-strategy-blend .blend-tooltip { position: absolute; pointer-events: none; z-index: 5; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); padding: 8px 10px; max-width: 280px; }
#equal-weight-strategy-blend .blend-tooltip div { white-space: nowrap; }
@media (max-width: 480px) {
  #equal-weight-strategy-blend .blend-metrics { gap: 8px 14px; }
  #equal-weight-strategy-blend .blend-panel-title, #equal-weight-strategy-blend .blend-note { margin-left: 62px; }
}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {
  const root = document.getElementById("equal-weight-strategy-blend");
  const raw = __DATA__;
  const parseDate = d3.utcParse("%Y-%m-%d");
  const data = raw.map(d => ({...d, date: parseDate(d.date)}));
  const defs = [
    {key:"equal_weight_blend", label:"50/50 等资金组合", color:"var(--viz-series-1)", width:2.9},
    {key:"strongest_candidate", label:"趋势专家 + CSI300 50% 对冲", color:"var(--viz-series-2)", width:1.8},
    {key:"formal_lgbm", label:"之前正式 LGBM", color:"var(--viz-series-3)", width:1.8}
  ];
  const visible = new Set(defs.map(d => d.key));
  const legend = d3.select(root).select(".blend-legend");
  const buttons = legend.selectAll("button").data(defs).join("button")
    .attr("type", "button").attr("aria-pressed", "true")
    .on("click", function(event, d) {
      if (visible.has(d.key) && visible.size > 1) visible.delete(d.key); else visible.add(d.key);
      d3.select(this).attr("aria-pressed", visible.has(d.key) ? "true" : "false");
      drawAll();
    });
  buttons.append("span").attr("class", "blend-swatch").style("--series-color", d => d.color);
  buttons.append("span").text(d => d.label);
  const tooltip = d3.select(root).select(".blend-tooltip");
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
    const yDomain = isDrawdown ? [Math.min(-0.01, extent[0] * 1.08), 0.015] : [Math.min(0.95, extent[0] * 0.98), extent[1] * 1.04];
    const y = d3.scaleLinear().domain(yDomain).nice().range([height - margin.bottom, margin.top]);
    svg.append("rect").attr("data-chart-frame", "").attr("x", margin.left).attr("y", margin.top)
      .attr("width", width - margin.left - margin.right).attr("height", height - margin.top - margin.bottom)
      .attr("fill", "none").attr("stroke", "var(--border)");
    svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(width < 480 ? 4 : 7).tickSizeOuter(0));
    svg.append("g").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(isDrawdown ? d3.format(".0%") : d => `${d.toFixed(1)}x`).tickSizeOuter(0));
    const line = field => d3.line().defined(d => Number.isFinite(d[field])).x(d => x(d.date)).y(d => y(d[field]));
    selected.forEach(def => {
      const field = isDrawdown ? `${def.key}_drawdown` : def.key;
      svg.append("path").datum(data).attr("fill", "none").attr("stroke", def.color).attr("stroke-width", def.width).attr("d", line(field)).attr("data-series", def.key);
    });
    const guide = svg.append("line").attr("data-chart-hover-guide", "").attr("y1", margin.top).attr("y2", height - margin.bottom).attr("stroke", "var(--border)").attr("pointer-events", "none").style("display", "none");
    const markers = svg.append("g").attr("pointer-events", "none");
    svg.append("rect").attr("data-chart-hit", "").attr("data-chart-hover-overlay", "cross-series")
      .attr("x", margin.left).attr("y", margin.top).attr("width", width - margin.left - margin.right).attr("height", height - margin.top - margin.bottom).attr("fill", "transparent")
      .on("pointermove", event => {
        const [px] = d3.pointer(event, svg.node());
        const clamped = Math.max(margin.left, Math.min(width - margin.right, px));
        const target = x.invert(clamped);
        guide.attr("x1", clamped).attr("x2", clamped).style("display", null);
        const rows = selected.map(def => ({def, value: valueAt(def.key, target, isDrawdown)}));
        markers.selectAll("circle").data(rows, d => d.def.key).join("circle").attr("data-chart-hover-marker", "").attr("cx", clamped).attr("cy", d => y(d.value)).attr("r", 4).attr("fill", d => d.def.color).attr("stroke", "var(--background)").attr("stroke-width", 1.5);
        tooltip.html(`<div><strong>${d3.utcFormat("%Y-%m-%d")(target)}</strong></div>` + rows.map(r => `<div>${r.def.label}: ${isDrawdown ? d3.format(".2%")(r.value) : r.value.toFixed(2) + "x"}</div>`).join(""));
        tooltip.attr("hidden", null).style("left", `${Math.min(root.clientWidth - 290, clamped + 12)}px`).style("top", `${event.clientY - root.getBoundingClientRect().top + 12}px`);
      })
      .on("pointerleave", () => { guide.style("display", "none"); markers.selectAll("circle").remove(); tooltip.attr("hidden", true); });
    svg.append("text").attr("class", "axis-title").attr("data-axis", "x").attr("x", (margin.left + width - margin.right) / 2).attr("y", height - 6).attr("text-anchor", "middle").text("交易日期");
    svg.append("text").attr("class", "axis-title").attr("data-axis", "y").attr("transform", `translate(15,${(margin.top + height - margin.bottom) / 2}) rotate(-90)`).attr("text-anchor", "middle").text(isDrawdown ? "回撤（%）" : "归一化净值（倍）");
  }
  function drawAll() { draw(".blend-main", false); draw(".blend-dd", true); }
  drawAll();
  new ResizeObserver(drawAll).observe(root);
})();
</script>
'''
    blend = summaries["equal_weight_blend"]
    return (
        template.replace("__DATA__", json.dumps(records, ensure_ascii=False, separators=(",", ":")))
        .replace("__TERMINAL__", f"{blend['terminal_value']:.2f}")
        .replace("__CAGR__", f"{blend['annual_return']:.2%}")
        .replace("__SHARPE__", f"{blend['sharpe']:.3f}")
        .replace("__MDD__", f"{blend['max_drawdown']:.2%}")
    )


def main() -> None:
    frame = build_frame()
    summaries = {name: performance(frame[name]) for name in SERIES}
    returns = frame[["strongest_candidate", "formal_lgbm"]].pct_change().dropna()
    correlation = float(returns.corr().iloc[0, 1])
    frame.to_csv(OUTPUT_CSV, encoding="utf-8-sig", float_format="%.10f")

    sampled = frame.iloc[::5].copy()
    if sampled.index[-1] != frame.index[-1]:
        sampled = pd.concat([sampled, frame.iloc[[-1]]])
    records = []
    for timestamp, row in sampled.iterrows():
        record = {"date": timestamp.date().isoformat()}
        for name in SERIES:
            record[name] = round(float(row[name]), 6)
            record[f"{name}_drawdown"] = round(float(row[f"{name}_drawdown"]), 6)
        records.append(record)
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html_fragment(records, summaries), encoding="utf-8")

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "historical_diagnostic_only_not_deployed",
        "window": [START, END],
        "construction": {
            "initial_weight_strongest_candidate": 0.50,
            "initial_weight_formal_lgbm": 0.50,
            "sleeve_rebalance": "none; each sleeve compounds independently",
            "formula": "blend_nav = 0.5 * strongest_nav + 0.5 * formal_lgbm_nav",
            "additional_cross_sleeve_transfer_cost": 0.0,
        },
        "daily_return_correlation": correlation,
        "metrics": summaries,
        "artifacts": {
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "png": str(OUTPUT_PNG.relative_to(ROOT)),
            "html": str(OUTPUT_HTML),
        },
        "production_change": False,
        "limitations": [
            "Both underlying curves are repeatedly revealed historical diagnostics.",
            "Underlying strategy returns already include their modeled stock and hedge costs.",
            "No additional cost is charged for transferring capital between sleeves because no sleeve rebalance is modeled.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "csv": str(OUTPUT_CSV),
        "json": str(OUTPUT_JSON),
        "html": str(OUTPUT_HTML),
        "metrics": summaries,
        "daily_return_correlation": correlation,
        "html_bytes": OUTPUT_HTML.stat().st_size,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
