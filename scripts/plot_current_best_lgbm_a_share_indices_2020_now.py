from __future__ import annotations

from datetime import datetime
import html
import json
from pathlib import Path
import sys
import tempfile

import httpx
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_current_best_lgbm_csi2000_2020_2026 import fetch_csi2000
from compare_current_best_lgbm_csi300_2020_2026 import (
    fetch_csi300,
    load_strategy,
    metrics,
)


ETF_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
ETF_SYMBOL = "sh563300"
OUTPUT_CSV = ROOT / "reports" / "current_best_lgbm_vs_csi300_csi2000_2020_now.csv"
OUTPUT_JSON = ROOT / "data" / "current_best_lgbm_vs_csi300_csi2000_2020_now.json"
OUTPUT_SVG = ROOT / "reports" / "current_best_lgbm_vs_csi300_csi2000_2020_now.svg"
OUTPUT_HTML = Path(tempfile.gettempdir()) / "lgbm-csi300-csi2000-2020-now.html"


def fetch_csi2000_etf(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, dict]:
    params = {
        "param": f"{ETF_SYMBOL},day,{start.date()},{end.date()},2000,qfq"
    }
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}
    with httpx.Client(timeout=30, trust_env=False, headers=headers) as client:
        response = client.get(ETF_URL, params=params)
        response.raise_for_status()
    payload = response.json()
    node = payload.get("data", {}).get(ETF_SYMBOL, {})
    rows = node.get("qfqday") or node.get("day") or []
    if len(rows) < 2:
        raise RuntimeError("CSI2000 ETF proxy has insufficient history")
    frame = pd.DataFrame(rows).iloc[:, :6]
    frame.columns = ["date", "open", "close", "high", "low", "volume"]
    frame["date"] = pd.to_datetime(frame["date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    values = (
        frame[["date", "close"]]
        .dropna()
        .drop_duplicates("date")
        .set_index("date")["close"]
        .sort_index()
    )
    return values, {
        "symbol": "563300",
        "name": "中证2000ETF国泰君安",
        "source": f"{ETF_URL}?param={ETF_SYMBOL},day,...",
        "fetched_at": datetime.now().astimezone().isoformat(),
    }


def extend_csi2000(
    official: pd.Series, etf: pd.Series, requested_end: pd.Timestamp
) -> tuple[pd.Series, dict]:
    overlap = pd.concat({"index": official, "etf": etf}, axis=1, join="inner").dropna()
    if len(overlap) < 100:
        raise RuntimeError("CSI2000 official/ETF overlap is insufficient for proxy validation")
    returns = overlap.pct_change().dropna()
    active = returns["etf"].sub(returns["index"])
    correlation = float(returns.corr().iloc[0, 1])
    tracking_error = float(active.std(ddof=0) * np.sqrt(252.0))
    relative_drift = float(
        (overlap["etf"].iloc[-1] / overlap["etf"].iloc[0])
        / (overlap["index"].iloc[-1] / overlap["index"].iloc[0])
        - 1.0
    )
    anchor = overlap.index.max()
    official_anchor = float(official.loc[anchor])
    etf_anchor = float(etf.loc[anchor])
    proxy = etf.loc[(etf.index > anchor) & (etf.index <= requested_end)].mul(
        official_anchor / etf_anchor
    )
    extended = pd.concat([official.loc[official.index <= anchor], proxy]).sort_index()
    extended = extended[~extended.index.duplicated(keep="last")]
    return extended, {
        "official_end": anchor.date().isoformat(),
        "proxy_start": proxy.index.min().date().isoformat() if not proxy.empty else None,
        "proxy_end": proxy.index.max().date().isoformat() if not proxy.empty else None,
        "overlap_start": overlap.index.min().date().isoformat(),
        "overlap_end": overlap.index.max().date().isoformat(),
        "overlap_days": int(len(overlap)),
        "daily_return_correlation": correlation,
        "annualized_tracking_error": tracking_error,
        "cumulative_relative_drift": relative_drift,
        "method": "Official CSI2000 through anchor; thereafter scale ETF qfq close returns from the same anchor.",
    }


def render_svg(curve: pd.DataFrame, proxy_start: pd.Timestamp) -> None:
    columns = ["current_best_lgbm", "csi2000", "csi300"]
    labels = {
        "current_best_lgbm": "Current best 75/25 LGBM",
        "csi2000": "CSI2000",
        "csi300": "CSI300",
    }
    colors = {
        "current_best_lgbm": "#176B87",
        "csi2000": "#8C5E24",
        "csi300": "#C2413B",
    }
    width, height = 1400, 800
    left, right, top, bottom = 105, 315, 105, 94
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
        '<text x="105" y="34" font-family="Segoe UI, Arial" font-size="24" font-weight="600" fill="#1F2933">LGBM vs CSI2000 vs CSI300, 2020 to latest</text>',
        f'<text x="105" y="65" font-family="Segoe UI, Arial" font-size="13" fill="#52606D">Common start {curve.index[0].date()} = 1.00 | CSI2000 uses ETF proxy after {proxy_start.date()}</text>',
    ]
    for tick in np.linspace(low, high, 6):
        yy = y(float(tick))
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_width}" y2="{yy:.2f}" stroke="#D9E2E8" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{yy + 4:.2f}" text-anchor="end" font-family="Segoe UI, Arial" font-size="12" fill="#52606D">{tick:.1f}x</text>')
    dates = pd.DatetimeIndex(curve.index)
    for year in range(dates.min().year, dates.max().year + 1):
        index = int(np.argmin(np.abs((dates - pd.Timestamp(year, 1, 1)).days)))
        parts.append(f'<text x="{x(index):.2f}" y="{height - 47}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="12" fill="#52606D">{year}</text>')
    for column in columns:
        if column != "csi2000":
            points = " ".join(f"{x(i):.2f},{y(float(v)):.2f}" for i, v in enumerate(curve[column]))
            parts.append(f'<polyline data-series="{column}" points="{points}" fill="none" stroke="{colors[column]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        else:
            official_mask = dates < proxy_start
            official_indices = np.flatnonzero(official_mask)
            proxy_indices = np.flatnonzero(dates >= proxy_start)
            if len(proxy_indices):
                proxy_indices = np.insert(proxy_indices, 0, max(proxy_indices[0] - 1, 0))
            for indices, dash in ((official_indices, ""), (proxy_indices, "9 6")):
                points = " ".join(f"{x(int(i)):.2f},{y(float(curve[column].iloc[int(i)])):.2f}" for i in indices)
                dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
                parts.append(f'<polyline data-series="{column}" points="{points}" fill="none" stroke="{colors[column]}" stroke-width="3"{dash_attr} stroke-linejoin="round" stroke-linecap="round"/>')
        value = float(curve[column].iloc[-1])
        yy = y(value)
        parts.append(f'<circle cx="{left + plot_width}" cy="{yy:.2f}" r="5" fill="{colors[column]}"/>')
        parts.append(f'<text x="{left + plot_width + 18}" y="{yy + 5:.2f}" font-family="Segoe UI, Arial" font-size="13" fill="#1F2933">{html.escape(labels[column])} {value:.2f}x</text>')
    parts.extend([
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#CBD5DC" stroke-width="1"/>',
        f'<text x="{left + plot_width / 2:.2f}" y="{height - 14}" text-anchor="middle" font-family="Segoe UI, Arial" font-size="13" fill="#52606D">A-share trading date</text>',
        f'<text x="25" y="{top + plot_height / 2:.2f}" transform="rotate(-90 25 {top + plot_height / 2:.2f})" text-anchor="middle" font-family="Segoe UI, Arial" font-size="13" fill="#52606D">Normalized net value (x)</text>',
        '</svg>',
    ])
    OUTPUT_SVG.write_text("\n".join(parts), encoding="utf-8")


def render_inline_html(curve: pd.DataFrame, proxy_start: pd.Timestamp) -> None:
    sample_indices = list(range(0, len(curve), 4))
    if sample_indices[-1] != len(curve) - 1:
        sample_indices.append(len(curve) - 1)
    sample = curve.iloc[sample_indices].reset_index()
    sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
    data = sample.round(6).to_dict(orient="records")
    fragment = f'''<section id="ashare-three-curve-viz">
  <h2>最强LGBM、中证2000与沪深300：2020至今</h2>
  <div class="viz-row" id="atc-legend" aria-label="曲线"></div>
  <div id="atc-chart"></div>
  <div class="tooltip" id="atc-tooltip" role="tooltip" hidden></div>
  <p class="text-small text-muted">中证2000在2025-12-31前使用官方指数；2026部分为中证2000ETF校准代理，以虚线表示。</p>
  <p class="sr-only" id="atc-summary">当前最强LGBM策略、中证2000和沪深300从2020年共同起点归一化后的净值曲线。</p>
</section>
<style>
  #ashare-three-curve-viz {{ position: relative; width: 100%; color: var(--foreground); }}
  #ashare-three-curve-viz h2 {{ margin: 0 0 8px; font-weight: 500; letter-spacing: 0; }}
  #ashare-three-curve-viz #atc-legend {{ margin-bottom: 8px; gap: 14px; }}
  #ashare-three-curve-viz .series-toggle {{ background: transparent; border: 0; color: var(--foreground); padding: 3px 0; display: inline-flex; align-items: center; gap: 6px; }}
  #ashare-three-curve-viz .series-toggle[aria-pressed="false"] {{ opacity: 0.48; }}
  #ashare-three-curve-viz .swatch {{ width: 20px; height: 3px; display: inline-block; background: var(--swatch); }}
  #ashare-three-curve-viz .swatch.proxy {{ height: 0; background: transparent; border-top: 3px dashed var(--swatch); }}
  #ashare-three-curve-viz #atc-chart {{ width: 100%; min-height: 380px; }}
  #ashare-three-curve-viz .plot {{ display: block; overflow: visible; }}
  #ashare-three-curve-viz .axis text, #ashare-three-curve-viz .axis-title, #ashare-three-curve-viz .end-label {{ fill: var(--foreground); font-size: 12px; }}
  #ashare-three-curve-viz .axis path, #ashare-three-curve-viz .axis line {{ stroke: var(--border); }}
  #ashare-three-curve-viz .grid line {{ stroke: var(--border); stroke-opacity: 0.55; }}
  #ashare-three-curve-viz .grid path {{ display: none; }}
  #ashare-three-curve-viz [data-chart-frame] {{ fill: transparent; stroke: var(--border); }}
  #ashare-three-curve-viz .tooltip {{ position: absolute; pointer-events: none; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); padding: 8px 10px; border-radius: 6px; font-size: 12px; z-index: 5; }}
  #ashare-three-curve-viz .tooltip-row {{ display: flex; justify-content: space-between; gap: 18px; white-space: nowrap; }}
  #ashare-three-curve-viz .tooltip-date {{ margin-bottom: 4px; font-weight: 500; }}
  #ashare-three-curve-viz > p {{ margin: 6px 0 0; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById("ashare-three-curve-viz");
  const chart = document.getElementById("atc-chart");
  const legend = document.getElementById("atc-legend");
  const tooltip = document.getElementById("atc-tooltip");
  const proxyStart = d3.timeParse("%Y-%m-%d")("{proxy_start.date().isoformat()}");
  const data = {json.dumps(data, ensure_ascii=True, separators=(',', ':'))};
  const series = [
    {{key:"current_best_lgbm",label:"最强75/25 LGBM",color:"var(--viz-series-1)"}},
    {{key:"csi2000",label:"中证2000",color:"var(--viz-series-2)",proxy:true}},
    {{key:"csi300",label:"沪深300",color:"var(--viz-series-3)"}}
  ];
  const parseDate = d3.timeParse("%Y-%m-%d");
  data.forEach(d => d.x = parseDate(d.date));
  const visible = new Set(series.map(d => d.key));
  series.forEach(item => {{
    const button = document.createElement("button");
    button.type = "button";
    button.className = "series-toggle";
    button.setAttribute("aria-pressed", "true");
    button.innerHTML = `<span class="swatch${{item.proxy ? " proxy" : ""}}" style="--swatch:${{item.color}}"></span><span>${{item.label}}</span>`;
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
    const height = width <= 400 ? 390 : 450;
    const margin = {{top:18,right:width <= 400 ? 16 : 152,bottom:54,left:68}};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const active = series.filter(s => visible.has(s.key));
    const extent = d3.extent(active.flatMap(s => data.map(d => d[s.key])));
    const padding = Math.max((extent[1] - extent[0]) * 0.06, 0.05);
    const x = d3.scaleTime().domain(d3.extent(data, d => d.x)).range([0, innerWidth]);
    const y = d3.scaleLinear().domain([Math.max(0, extent[0] - padding), extent[1] + padding]).nice().range([innerHeight, 0]);
    const svg = d3.select(chart).append("svg").attr("class","plot").attr("viewBox",`0 0 ${{width}} ${{height}}`).attr("width",width).attr("height",height).attr("role","img").attr("aria-labelledby","atc-chart-title atc-chart-desc");
    svg.append("title").attr("id","atc-chart-title").text("最强LGBM、中证2000与沪深300");
    svg.append("desc").attr("id","atc-chart-desc").text(document.getElementById("atc-summary").textContent);
    const g = svg.append("g").attr("transform",`translate(${{margin.left}},${{margin.top}})`);
    g.append("rect").attr("data-chart-frame","").attr("width",innerWidth).attr("height",innerHeight);
    g.append("g").attr("class","grid").call(d3.axisLeft(y).ticks(6).tickSize(-innerWidth).tickFormat(""));
    g.append("g").attr("class","axis").attr("transform",`translate(0,${{innerHeight}})`).call(d3.axisBottom(x).ticks(width <= 400 ? 4 : 7).tickFormat(d3.timeFormat("%Y")));
    g.append("g").attr("class","axis").call(d3.axisLeft(y).ticks(6).tickFormat(d => `${{d.toFixed(1)}}x`));
    g.append("text").attr("class","axis-title").attr("data-axis","x").attr("x",innerWidth/2).attr("y",innerHeight+44).attr("text-anchor","middle").text("A股交易日期");
    g.append("text").attr("class","axis-title").attr("data-axis","y").attr("transform","rotate(-90)").attr("x",-innerHeight/2).attr("y",-52).attr("text-anchor","middle").text("归一化净值（倍）");
    const line = key => d3.line().x(d => x(d.x)).y(d => y(d[key]));
    active.forEach(item => {{
      if (item.key === "csi2000") {{
        const official = data.filter(d => d.x < proxyStart);
        const proxy = data.filter(d => d.x >= proxyStart);
        if (proxy.length && official.length) proxy.unshift(official[official.length - 1]);
        g.append("path").attr("data-series",item.key).attr("fill","none").attr("stroke",item.color).attr("stroke-width",2.4).attr("d",line(item.key)(official));
        g.append("path").attr("data-series",item.key).attr("data-proxy-segment","").attr("fill","none").attr("stroke",item.color).attr("stroke-width",2.4).attr("stroke-dasharray","8 5").attr("d",line(item.key)(proxy));
      }} else {{
        g.append("path").attr("data-series",item.key).attr("fill","none").attr("stroke",item.color).attr("stroke-width",2.4).attr("d",line(item.key)(data));
      }}
      if (width > 400) {{ const end=data[data.length-1]; g.append("text").attr("class","end-label").attr("x",innerWidth+9).attr("y",y(end[item.key])+4).text(`${{item.label}} ${{end[item.key].toFixed(2)}}x`); }}
    }});
    const guide=g.append("line").attr("data-chart-hover-guide","").attr("y1",0).attr("y2",innerHeight).attr("stroke","var(--foreground)").attr("stroke-opacity",0.35).style("display","none");
    const markers=new Map(active.map(item=>[item.key,g.append("circle").attr("data-chart-hover-marker",item.key).attr("r",4).attr("fill",item.color).style("display","none")]));
    const bisect=d3.bisector(d=>d.x).center;
    g.append("rect").attr("data-chart-hit","").attr("data-chart-hover-overlay","cross-series").attr("width",innerWidth).attr("height",innerHeight).attr("fill","transparent")
      .on("pointermove",event=>{{ const [px]=d3.pointer(event); const target=x.invert(px); const index=bisect(data,target); const right=data[Math.max(0,Math.min(data.length-1,index))]; const left=data[Math.max(0,index-1)]; const span=Math.max(right.x-left.x,1); const ratio=Math.max(0,Math.min(1,(target-left.x)/span)); const interpolated=Object.fromEntries(active.map(item=>[item.key,left[item.key]+(right[item.key]-left[item.key])*ratio])); const gx=x(target); guide.attr("x1",gx).attr("x2",gx).style("display",null); active.forEach(item=>markers.get(item.key).attr("cx",gx).attr("cy",y(interpolated[item.key])).style("display",null)); tooltip.innerHTML=`<div class="tooltip-date">${{d3.timeFormat("%Y-%m-%d")(target)}}</div>`+active.map(item=>`<div class="tooltip-row"><span>${{item.label}}${{item.key === "csi2000" && target >= proxyStart ? "（代理）" : ""}}</span><strong>${{interpolated[item.key].toFixed(2)}}x</strong></div>`).join(""); tooltip.hidden=false; tooltip.style.left=`${{Math.max(0,Math.min(root.clientWidth-220,margin.left+gx+12))}}px`; tooltip.style.top=`${{Math.max(42,margin.top+8)}}px`; }})
      .on("pointerleave",()=>{{ guide.style("display","none"); markers.forEach(marker=>marker.style("display","none")); tooltip.hidden=true; }});
  }}
  draw();
  new ResizeObserver(draw).observe(chart);
}})();
</script>
'''
    OUTPUT_HTML.write_text(fragment, encoding="utf-8")


def main() -> None:
    strategy = load_strategy()
    csi300, csi300_source = fetch_csi300(strategy.index.min(), strategy.index.max())
    csi2000_official, csi2000_source = fetch_csi2000(
        strategy.index.min(), strategy.index.max()
    )
    etf, etf_source = fetch_csi2000_etf(pd.Timestamp("2023-01-01"), strategy.index.max())
    csi2000, proxy_audit = extend_csi2000(
        csi2000_official, etf, strategy.index.max()
    )
    common_dates = strategy.index.intersection(csi300.index).intersection(csi2000.index).sort_values()
    curve = pd.DataFrame(
        {
            "current_best_lgbm": strategy.reindex(common_dates),
            "csi2000": csi2000.reindex(common_dates),
            "csi300": csi300.reindex(common_dates),
        },
        index=common_dates,
    ).dropna()
    curve = curve.div(curve.iloc[0])
    curve.index.name = "date"
    curve.to_csv(OUTPUT_CSV, encoding="utf-8-sig", float_format="%.10f")
    proxy_start = pd.Timestamp(proxy_audit["proxy_start"])
    render_svg(curve, proxy_start)
    render_inline_html(curve, proxy_start)

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "window": {
            "start": curve.index[0].date().isoformat(),
            "end": curve.index[-1].date().isoformat(),
            "trading_days": int(len(curve)),
        },
        "curves": {
            "current_best_lgbm": {
                "name": "Current best 75/25 LGBM",
                "metrics": metrics(curve["current_best_lgbm"]),
                "source": str((ROOT / "reports" / "overnight_factor_lgbm_2020_2026_curves.csv").relative_to(ROOT)),
            },
            "csi2000": {
                "name": "中证2000",
                "symbol": "932000",
                "metrics": metrics(curve["csi2000"]),
                "official_source": csi2000_source,
                "proxy_source": etf_source,
                "proxy_audit": proxy_audit,
            },
            "csi300": {
                "name": "沪深300",
                "symbol": "000300",
                "metrics": metrics(curve["csi300"]),
                "source": csi300_source,
            },
        },
        "artifacts": {
            "csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "json": str(OUTPUT_JSON.relative_to(ROOT)),
            "svg": str(OUTPUT_SVG.relative_to(ROOT)),
            "inline_html": str(OUTPUT_HTML),
        },
        "limitations": [
            "CSI2000 after 2025-12-31 is an ETF-calibrated proxy, not an official index observation.",
            "All index curves are price-return curves and exclude dividends.",
            "The strategy uses a backfilled current Top50 universe and the revealed period is not an untouched holdout.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
