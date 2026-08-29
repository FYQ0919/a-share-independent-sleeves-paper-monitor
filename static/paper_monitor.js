const monitorState = { chart: null, payload: null };
const $ = selector => document.querySelector(selector);
const money = value => Number(value || 0).toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const pct = value => `${Number(value || 0) >= 0 ? "+" : ""}${(Number(value || 0) * 100).toFixed(2)}%`;
const plainPct = value => `${(Number(value || 0) * 100).toFixed(1)}%`;
const signedClass = value => Number(value || 0) >= 0 ? "positive" : "negative";

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(window.monitorToastTimer);
  window.monitorToastTimer = setTimeout(() => node.classList.remove("show"), 2800);
}

async function request(url) {
  const response = await fetch(url, { headers: { "Accept": "application/json" } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `请求失败 (${response.status})`);
  return payload;
}

function renderPositions(selector, positions) {
  const body = $(selector);
  if (!positions?.length) {
    body.innerHTML = '<tr><td colspan="5" class="monitor-empty-cell">等待下一交易日开盘建仓</td></tr>';
    return;
  }
  body.innerHTML = positions.map(item => `
    <tr>
      <td><strong>${item.name || item.code}</strong><small>${item.code}</small></td>
      <td>${Number(item.shares || 0).toLocaleString("zh-CN")}股</td>
      <td>¥${money(item.last_price)}</td>
      <td>¥${money(item.market_value)}</td>
      <td class="${signedClass(item.unrealized_pnl)}">${money(item.unrealized_pnl)}</td>
    </tr>`).join("");
}

function orderText(decision) {
  if (!decision) return ["尚无决策", "--"];
  const pending = decision.pending_order;
  if (!pending) return [decision.action_label || "继续持有", decision.reason || "本日无交易"];
  const buys = (pending.buys || []).map(item => item.name || item.code).join("、") || "无新增";
  return [`${pending.action_label || "待执行"} · 下一交易日开盘`, `买入：${buys}`];
}

function renderChart(history) {
  const chartNode = $("#monitor-equity-chart");
  const empty = $("#monitor-chart-empty");
  if (!history?.length) {
    chartNode.hidden = true;
    empty.hidden = false;
    return;
  }
  chartNode.hidden = false;
  empty.hidden = true;
  if (!monitorState.chart) monitorState.chart = echarts.init(chartNode, null, { renderer: "canvas" });
  const dates = history.map(item => item.date);
  const common = {
    type: "line", showSymbol: history.length < 16, symbolSize: 6, smooth: false,
    connectNulls: true, emphasis: { focus: "series" },
  };
  monitorState.chart.setOption({
    animationDuration: 280,
    color: ["#217f5b", "#176b87", "#c2413b", "#7b8791"],
    tooltip: { trigger: "axis", axisPointer: { type: "line" }, valueFormatter: value => Number(value).toFixed(4) },
    legend: { top: 0, left: 0, itemWidth: 18, itemHeight: 3, textStyle: { color: "#34443b" } },
    grid: [
      { left: 66, right: 24, top: 46, height: "58%" },
      { left: 66, right: 24, top: "76%", height: "14%" },
    ],
    xAxis: [
      { type: "category", data: dates, boundaryGap: false, axisLabel: { hideOverlap: true }, axisLine: { lineStyle: { color: "#cfd8d2" } } },
      { type: "category", gridIndex: 1, data: dates, boundaryGap: false, axisLabel: { hideOverlap: true }, axisLine: { lineStyle: { color: "#cfd8d2" } } },
    ],
    yAxis: [
      { type: "value", scale: true, name: "净值 (x)", nameLocation: "middle", nameGap: 48, splitLine: { lineStyle: { color: "#e2e8e4" } } },
      { type: "value", gridIndex: 1, max: 0, axisLabel: { formatter: value => `${(value * 100).toFixed(1)}%` }, splitLine: { lineStyle: { color: "#e2e8e4" } } },
    ],
    dataZoom: history.length > 80 ? [{ type: "inside", xAxisIndex: [0, 1], start: 0, end: 100 }] : [],
    series: [
      { ...common, name: "组合", data: history.map(item => item.equity), lineStyle: { width: 3 } },
      { ...common, name: "趋势75%", data: history.map(item => item.trend_equity), lineStyle: { width: 2 } },
      { ...common, name: "LGBM 25%", data: history.map(item => item.lgbm_equity), lineStyle: { width: 2 } },
      { ...common, name: "组合回撤", xAxisIndex: 1, yAxisIndex: 1, data: history.map(item => item.drawdown), areaStyle: { opacity: 0.12 }, lineStyle: { width: 1.5 } },
    ],
  }, true);
}

function render(payload) {
  monitorState.payload = payload;
  const snapshot = payload.snapshot;
  const status = $("#monitor-status");
  status.querySelector("span:last-child").textContent = payload.enabled ? (snapshot ? "前向运行中" : "等待首个信号") : "未启用";
  status.classList.toggle("is-waiting", !snapshot);
  $("#monitor-account-id").textContent = payload.account_id || "独立袖套协议尚未建立";
  const observations = payload.history?.length || 0;
  const target = payload.config?.promotion_required_sessions || 126;
  const progress = Math.min(100, observations / target * 100);
  $("#monitor-progress-label").textContent = `${observations} / ${target}`;
  $("#monitor-progress-bar").style.width = `${progress}%`;
  $(".monitor-progress").setAttribute("aria-valuenow", String(observations));
  $("#monitor-latest-date").textContent = snapshot ? `最新信号 ${snapshot.signal_date}` : `前向起点 ${payload.config?.forward_start_after || "--"} 之后`;

  if (!snapshot) {
    renderChart(payload.history || []);
    return;
  }
  const trend = snapshot.sleeves?.trend || {};
  const lgbm = snapshot.sleeves?.lgbm || {};
  $("#monitor-nav").textContent = money(snapshot.nav);
  $("#monitor-return").textContent = pct(snapshot.cumulative_return);
  $("#monitor-return").className = signedClass(snapshot.cumulative_return);
  $("#monitor-daily-pnl").textContent = money(snapshot.daily_pnl);
  $("#monitor-daily-pnl").className = signedClass(snapshot.daily_pnl);
  $("#monitor-drawdown").textContent = pct(snapshot.drawdown);
  $("#monitor-drawdown").className = Number(snapshot.drawdown) < 0 ? "negative" : "";
  $("#monitor-cost").textContent = money(snapshot.transaction_cost_total);
  $("#monitor-weight-drift").textContent = `实际 ${plainPct(trend.actual_weight)} / ${plainPct(lgbm.actual_weight)}`;

  $("#trend-actual-weight").textContent = plainPct(trend.actual_weight);
  $("#trend-nav").textContent = `¥${money(trend.nav)}`;
  $("#trend-return").textContent = pct(trend.cumulative_return);
  $("#trend-return").className = signedClass(trend.cumulative_return);
  $("#trend-exposure").textContent = plainPct(trend.exposure);
  $("#trend-regime").textContent = trend.regime?.strong_equipment ? "设备强势" : "普通环境";
  const trendOrder = orderText(trend.decision);
  $("#trend-action").textContent = trendOrder[0];
  $("#trend-order").textContent = trendOrder[1];
  renderPositions("#trend-positions", trend.positions);

  $("#lgbm-actual-weight").textContent = plainPct(lgbm.actual_weight);
  $("#lgbm-nav").textContent = `¥${money(lgbm.nav)}`;
  $("#lgbm-return").textContent = pct(lgbm.cumulative_return);
  $("#lgbm-return").className = signedClass(lgbm.cumulative_return);
  $("#lgbm-exposure").textContent = plainPct(lgbm.exposure);
  $("#lgbm-cycle").textContent = `${Number(lgbm.decision?.cycle_day || 0)} / ${payload.config?.rebalance_days || 10}`;
  const lgbmOrder = orderText(lgbm.decision);
  $("#lgbm-action").textContent = lgbmOrder[0];
  $("#lgbm-order").textContent = lgbmOrder[1];
  renderPositions("#lgbm-positions", lgbm.positions);

  $("#hedge-action").textContent = trend.hedge_action || "--";
  $("#hedge-index-close").textContent = money(trend.index_close);
  $("#hedge-index-ma").textContent = money(trend.index_ma);
  $("#hedge-active").textContent = plainPct(trend.active_hedge_ratio);
  $("#hedge-target").textContent = plainPct(trend.target_hedge_ratio);
  $("#hedge-maximum").textContent = plainPct(payload.config?.trend_maximum_hedge);
  const maxHedge = Number(payload.config?.trend_maximum_hedge || 0.75);
  $("#hedge-scale-value").style.width = `${Math.min(100, Number(trend.active_hedge_ratio || 0) / maxHedge * 100)}%`;
  renderChart(payload.history || []);
}

async function loadMonitor(showToast = false) {
  const button = $("#monitor-refresh");
  button.disabled = true;
  try {
    render(await request("/api/paper-monitor?limit=500"));
    if (showToast) toast("模拟盘已刷新");
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) window.lucide.createIcons();
  $("#monitor-refresh").addEventListener("click", () => loadMonitor(true));
  window.addEventListener("resize", () => monitorState.chart?.resize());
  await loadMonitor(false);
});
