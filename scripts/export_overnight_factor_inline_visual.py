from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "overnight_factor_lgbm_2020_2026_curves.csv"
OUTPUT = ROOT / "reports" / "overnight-factor-lgbm-2020-2026.html"

SERIES = {
    "previous_lgbm": "Previous LGBM",
    "lgbm_factor_feature": "LGBM + factor feature",
    "lgbm_factor_overlay": "LGBM + fixed 15% overlay",
}


def main() -> None:
    frame = pd.read_csv(SOURCE, parse_dates=["date"])
    keep = list(range(0, len(frame), 5))
    if keep[-1] != len(frame) - 1:
        keep.append(len(frame) - 1)
    sampled = frame.iloc[keep][["date", *SERIES]].copy()
    sampled["date"] = sampled["date"].dt.strftime("%Y-%m-%d")
    data = sampled.round(6).to_dict(orient="records")
    meta = [
        {"key": key, "label": label, "color": f"var(--viz-series-{index})"}
        for index, (key, label) in enumerate(SERIES.items(), start=1)
    ]
    fragment = f'''<section id="overnight-factor-lgbm-viz">
  <h2>Overnight factor LGBM comparison, 2020-2026</h2>
  <div class="viz-row" id="ofl-legend" aria-label="Strategy series"></div>
  <div id="ofl-chart"></div>
  <div class="tooltip" id="ofl-tooltip" role="tooltip" hidden></div>
  <p class="sr-only" id="ofl-summary">Normalized net value for three strategies from January 2020 through August 2026. Previous LGBM finishes highest.</p>
</section>
<style>
  #overnight-factor-lgbm-viz {{ width: 100%; color: var(--foreground); }}
  #overnight-factor-lgbm-viz h2 {{ margin: 0 0 8px; font-weight: 500; letter-spacing: 0; }}
  #overnight-factor-lgbm-viz #ofl-legend {{ margin-bottom: 8px; gap: 14px; }}
  #overnight-factor-lgbm-viz .series-toggle {{ background: transparent; border: 0; color: var(--foreground); padding: 3px 0; display: inline-flex; align-items: center; gap: 6px; }}
  #overnight-factor-lgbm-viz .series-toggle[aria-pressed="false"] {{ opacity: 0.48; }}
  #overnight-factor-lgbm-viz .swatch {{ width: 20px; height: 3px; display: inline-block; background: var(--swatch); }}
  #overnight-factor-lgbm-viz #ofl-chart {{ width: 100%; min-height: 360px; }}
  #overnight-factor-lgbm-viz .plot {{ display: block; overflow: visible; }}
  #overnight-factor-lgbm-viz .axis text, #overnight-factor-lgbm-viz .axis-title, #overnight-factor-lgbm-viz .end-label {{ fill: var(--foreground); font-size: 12px; }}
  #overnight-factor-lgbm-viz .axis path, #overnight-factor-lgbm-viz .axis line {{ stroke: var(--border); }}
  #overnight-factor-lgbm-viz .grid line {{ stroke: var(--border); stroke-opacity: 0.55; }}
  #overnight-factor-lgbm-viz .grid path {{ display: none; }}
  #overnight-factor-lgbm-viz [data-chart-frame] {{ fill: transparent; stroke: var(--border); }}
  #overnight-factor-lgbm-viz .tooltip {{ position: absolute; pointer-events: none; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); padding: 8px 10px; border-radius: 6px; font-size: 12px; z-index: 5; }}
  #overnight-factor-lgbm-viz .tooltip-row {{ display: flex; justify-content: space-between; gap: 18px; white-space: nowrap; }}
  #overnight-factor-lgbm-viz .tooltip-date {{ margin-bottom: 4px; font-weight: 500; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById("overnight-factor-lgbm-viz");
  const chart = document.getElementById("ofl-chart");
  const legend = document.getElementById("ofl-legend");
  const tooltip = document.getElementById("ofl-tooltip");
  const data = {json.dumps(data, ensure_ascii=True, separators=(',', ':'))};
  const series = {json.dumps(meta, ensure_ascii=True, separators=(',', ':'))};
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
      if (visible.has(item.key) && visible.size > 1) visible.delete(item.key);
      else visible.add(item.key);
      button.setAttribute("aria-pressed", String(visible.has(item.key)));
      draw();
    }});
    legend.appendChild(button);
  }});

  function draw() {{
    chart.replaceChildren();
    const width = Math.max(320, chart.getBoundingClientRect().width || 736);
    const height = width <= 400 ? 370 : 430;
    const margin = {{top: 18, right: width <= 400 ? 16 : 170, bottom: 54, left: 68}};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const active = series.filter(s => visible.has(s.key));
    const allValues = active.flatMap(s => data.map(d => d[s.key]));
    const extent = d3.extent(allValues);
    const padding = Math.max((extent[1] - extent[0]) * 0.06, 0.05);
    const x = d3.scaleTime().domain(d3.extent(data, d => d.x)).range([0, innerWidth]);
    const y = d3.scaleLinear().domain([Math.max(0, Math.min(0.9, extent[0] - padding)), extent[1] + padding]).nice().range([innerHeight, 0]);
    const svg = d3.select(chart).append("svg")
      .attr("class", "plot")
      .attr("viewBox", `0 0 ${{width}} ${{height}}`)
      .attr("width", width)
      .attr("height", height)
      .attr("role", "img")
      .attr("aria-labelledby", "ofl-chart-title ofl-chart-desc");
    svg.append("title").attr("id", "ofl-chart-title").text("LGBM strategy normalized net value comparison");
    svg.append("desc").attr("id", "ofl-chart-desc").text(document.getElementById("ofl-summary").textContent);
    const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);
    g.append("rect").attr("data-chart-frame", "").attr("width", innerWidth).attr("height", innerHeight);
    g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(6).tickSize(-innerWidth).tickFormat(""));
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${{innerHeight}})`).call(d3.axisBottom(x).ticks(width <= 400 ? 4 : 7).tickFormat(d3.timeFormat("%Y")));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6).tickFormat(d => `${{d.toFixed(1)}}x`));
    g.append("text").attr("class", "axis-title").attr("data-axis", "x").attr("x", innerWidth / 2).attr("y", innerHeight + 44).attr("text-anchor", "middle").text("Signal date");
    g.append("text").attr("class", "axis-title").attr("data-axis", "y").attr("transform", "rotate(-90)").attr("x", -innerHeight / 2).attr("y", -52).attr("text-anchor", "middle").text("Normalized net value (x)");
    const line = key => d3.line().x(d => x(d.x)).y(d => y(d[key]))(data);
    active.forEach((item, index) => {{
      g.append("path").attr("data-series", item.key).attr("fill", "none").attr("stroke", item.color).attr("stroke-width", 2.3).attr("stroke-dasharray", index === 1 ? "8 5" : index === 2 ? "3 4" : null).attr("d", line(item.key));
      if (width > 400) {{
        const end = data[data.length - 1];
        g.append("text").attr("class", "end-label").attr("x", innerWidth + 9).attr("y", y(end[item.key]) + 4).text(`${{item.label}} ${{end[item.key].toFixed(2)}}x`);
      }}
    }});
    const guide = g.append("line").attr("data-chart-hover-guide", "").attr("y1", 0).attr("y2", innerHeight).attr("stroke", "var(--foreground)").attr("stroke-opacity", 0.35).style("display", "none");
    const markers = new Map(active.map(item => [item.key, g.append("circle").attr("data-chart-hover-marker", item.key).attr("r", 4).attr("fill", item.color).style("display", "none")]));
    const bisect = d3.bisector(d => d.x).center;
    g.append("rect")
      .attr("data-chart-hit", "")
      .attr("data-chart-hover-overlay", "cross-series")
      .attr("width", innerWidth)
      .attr("height", innerHeight)
      .attr("fill", "transparent")
      .on("pointermove", event => {{
        const [px] = d3.pointer(event);
        const target = x.invert(px);
        const index = bisect(data, target);
        const row = data[Math.max(0, Math.min(data.length - 1, index))];
        const gx = x(target);
        guide.attr("x1", gx).attr("x2", gx).style("display", null);
        active.forEach(item => markers.get(item.key).attr("cx", x(row.x)).attr("cy", y(row[item.key])).style("display", null));
        tooltip.innerHTML = `<div class="tooltip-date">${{row.date}}</div>` + active.map(item => `<div class="tooltip-row"><span>${{item.label}}</span><strong>${{row[item.key].toFixed(2)}}x</strong></div>`).join("");
        tooltip.hidden = false;
        tooltip.style.left = `${{Math.min(root.clientWidth - 210, margin.left + gx + 12)}}px`;
        tooltip.style.top = `${{Math.max(42, margin.top + 8)}}px`;
      }})
      .on("pointerleave", () => {{
        guide.style("display", "none");
        markers.forEach(marker => marker.style("display", "none"));
        tooltip.hidden = true;
      }});
  }}
  draw();
  new ResizeObserver(draw).observe(chart);
}})();
</script>
'''
    OUTPUT.write_text(fragment, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
