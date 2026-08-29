const btState = { mode: "demo", latest: null, history: [], chart: null };
const $ = selector => document.querySelector(selector);
const percent = value => `${Number(value || 0) >= 0 ? "+" : ""}${(Number(value || 0) * 100).toFixed(2)}%`;
const number = (value, digits = 2) => Number(value || 0).toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const escapeHtml = value => String(value ?? "").replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);

function toast(message) {
  const node = $("#toast");
  node.textContent = message; node.classList.add("show");
  clearTimeout(window.btToast); window.btToast = setTimeout(() => node.classList.remove("show"), 3500);
}

async function request(url, options = {}) {
  const response = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `请求失败 (${response.status})`);
  return data;
}

function codesFromInput() {
  return $("#stock-codes").value.split(/[\s,，;；]+/).map(item => item.trim()).filter(Boolean);
}

function updateUniverseCount() { $("#universe-count").textContent = `${new Set(codesFromInput()).size} 只`; }

function updateAdaptiveFields() {
  const adaptive = $("#rebalance-policy").value === "adaptive";
  document.querySelectorAll(".adaptive-only").forEach(node => { node.hidden = !adaptive; });
}

function metricClass(value, inverse = false) {
  const positive = Number(value) >= 0;
  return (inverse ? !positive : positive) ? "positive" : "negative";
}

function renderChart(result) {
  if (!window.echarts) return;
  if (!btState.chart) btState.chart = echarts.init($("#equity-chart"));
  const dates = result.curves.map(item => item.date);
  btState.chart.setOption({
    animationDuration: 450,
    textStyle: { fontFamily: '"Noto Sans SC", sans-serif', color: "#69716d" },
    tooltip: { trigger: "axis", valueFormatter: value => number(value, 4) },
    legend: { top: 10, right: 8, data: ["策略净值", "等权基准"] },
    grid: [{ left: 48, right: 22, top: 50, height: "55%" }, { left: 48, right: 22, top: "73%", height: "17%" }],
    xAxis: [
      { type: "category", data: dates, boundaryGap: false, axisLabel: { hideOverlap: true }, axisLine: { lineStyle: { color: "#dfe3e0" } } },
      { type: "category", data: dates, boundaryGap: false, gridIndex: 1, axisLabel: { hideOverlap: true }, axisLine: { lineStyle: { color: "#dfe3e0" } } },
    ],
    yAxis: [
      { type: "value", scale: true, splitLine: { lineStyle: { color: "#edf0ee" } } },
      { type: "value", gridIndex: 1, axisLabel: { formatter: value => `${(value * 100).toFixed(0)}%` }, splitLine: { lineStyle: { color: "#edf0ee" } } },
    ],
    dataZoom: [{ type: "inside", xAxisIndex: [0, 1] }, { type: "slider", xAxisIndex: [0, 1], bottom: 0, height: 17 }],
    series: [
      { name: "策略净值", type: "line", data: result.curves.map(item => item.strategy), showSymbol: false, lineStyle: { width: 2, color: "#16724a" }, itemStyle: { color: "#16724a" }, xAxisIndex: 0, yAxisIndex: 0 },
      { name: "等权基准", type: "line", data: result.curves.map(item => item.benchmark), showSymbol: false, lineStyle: { width: 1.5, color: "#a97818" }, itemStyle: { color: "#a97818" }, xAxisIndex: 0, yAxisIndex: 0 },
      { name: "回撤", type: "line", data: result.curves.map(item => item.drawdown), showSymbol: false, lineStyle: { width: 1, color: "#b8453c" }, areaStyle: { color: "rgba(184,69,60,.16)" }, xAxisIndex: 1, yAxisIndex: 1 },
    ],
  }, true);
}

function renderPerformance(metrics, data = {}) {
  const items = [
    ["基准收益", percent(metrics.benchmark_return)], ["基准年化", percent(metrics.benchmark_annual_return)], ["超额收益", percent(metrics.excess_return)],
    ["年化波动", percent(metrics.annual_volatility)], ["Sortino", number(metrics.sortino)],
    ["Calmar", number(metrics.calmar)], ["胜率", percent(metrics.win_rate)],
    ["年化 Alpha", percent(metrics.alpha)], ["Beta", number(metrics.beta)],
    ["信息比率", number(metrics.information_ratio)], ["平均换手", percent(metrics.average_turnover)],
    ["累计成本", percent(metrics.total_cost)], ["调仓次数", String(metrics.rebalance_count)],
    ["熔断次数", String(data.risk_halt_count ?? 0)],
  ];
  $("#performance-grid").innerHTML = items.map(([label, value]) => `<div class="performance-item"><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function renderRebalances(records) {
  $("#rebalance-body").innerHTML = records.map(record => `
    <tr><td>${record.signal_date}</td><td>${record.execution_date}</td><td><strong>${record.trigger === "scheduled" || !record.trigger ? "主调仓" : record.trigger === "risk" ? "风险退出" : "提前换股"}</strong><span class="stock-code">${escapeHtml(record.reason || "固定周期")}</span></td><td><div class="holding-tags">${record.holdings.map(item => `<span class="holding-tag">${escapeHtml(item.name)} ${escapeHtml(item.code)} · ${item.score.toFixed(1)}</span>`).join("")}</div></td><td>${record.holdings.length ? percent(record.holdings[0].weight) : "--"}</td></tr>`).join("");
  const adaptiveCount = records.filter(record => record.trigger && record.trigger !== "scheduled").length;
  $("#rebalance-summary").textContent = `${records.length} 次执行${adaptiveCount ? ` · ${adaptiveCount} 次提前` : ""}`;
}

function renderResult(result) {
  if (!result) return;
  btState.latest = result;
  btState.mode = result.config.mode;
  document.querySelectorAll(".segment").forEach(button => button.classList.toggle("active", button.dataset.mode === btState.mode));
  $("#start-date").value = result.config.start_date;
  $("#end-date").value = result.config.end_date;
  $("#top-n").value = result.config.top_n;
  $("#rebalance-days").value = result.config.rebalance_days;
  $("#initial-capital").value = result.config.initial_capital;
  $("#cost-bps").value = result.config.cost_bps;
  $("#strategy").value = result.config.strategy || "multi_factor";
  $("#rebalance-policy").value = result.config.rebalance_policy || "fixed";
  $("#min-hold-days").value = result.config.min_hold_days ?? 5;
  $("#rank-buffer").value = result.config.rank_buffer ?? 20;
  $("#score-gap").value = result.config.score_gap ?? 15;
  $("#risk-overlay").value = result.config.risk_overlay || "trend_volatility";
  $("#drawdown-limit").value = String(result.config.drawdown_limit ?? 0.25);
  $("#drawdown-cooldown").value = result.config.drawdown_cooldown_days ?? 10;
  updateAdaptiveFields();
  $("#stock-codes").value = result.config.codes.join(", ");
  updateUniverseCount();
  const metrics = result.metrics, capital = result.config.initial_capital;
  $("#bt-total-return").textContent = percent(metrics.total_return);
  $("#bt-total-return").className = metricClass(metrics.total_return);
  $("#bt-annual-return").textContent = percent(metrics.annual_return);
  $("#bt-annual-return").className = metricClass(metrics.annual_return);
  $("#bt-max-drawdown").textContent = percent(metrics.max_drawdown);
  $("#bt-max-drawdown").className = "negative";
  $("#bt-sharpe").textContent = number(metrics.sharpe);
  $("#bt-final-capital").textContent = `${number(capital * (1 + metrics.total_return), 0)} 元`;
  $("#bt-period").textContent = `${result.config.start_date} 至 ${result.config.end_date} · ${result.data.trading_days} 个交易日`;
  const overlayLabel = result.config.risk_overlay === "none" ? "无回撤覆盖" : result.config.risk_overlay === "balanced" ? "均衡覆盖" : result.config.risk_overlay === "adaptive" ? "自适应覆盖" : "保守覆盖";
  const circuitLabel = Number(result.config.drawdown_limit || 0) > 0 ? `熔断 ${(Number(result.config.drawdown_limit) * 100).toFixed(0)}%` : "无熔断";
  $("#execution-label").textContent = `${result.data.strategy || "原始多因子"} · ${result.data.policy_label || "固定周期"} · ${overlayLabel} · ${circuitLabel} · ${result.data.execution}`;
  $("#backtest-status").textContent = result.config.mode === "live" ? "真实历史" : "演示回测";
  const warning = $("#backtest-warnings"); warning.hidden = !result.warnings.length; warning.textContent = result.warnings.join("；");
  renderChart(result); renderPerformance(metrics, result.data); renderRebalances(result.rebalances);
}

function renderHistory(items) {
  btState.history = items;
  $("#backtest-history").innerHTML = items.map((item, index) => `<button class="backtest-history-item" data-index="${index}"><strong>${item.config.start_date} → ${item.config.end_date}</strong><span>${item.data.strategy || "原始多因子"} · ${item.data.policy_label || "固定周期"} · ${item.config.mode === "live" ? "真实历史" : "演示数据"} · Top ${item.config.top_n}</span><span>累计 ${percent(item.metrics.total_return)} · 超额 ${percent(item.metrics.excess_return)} · 回撤 ${percent(item.metrics.max_drawdown)}</span></button>`).join("") || '<p class="subtle">暂无历史回测</p>';
  document.querySelectorAll(".backtest-history-item").forEach(button => button.addEventListener("click", () => renderResult(btState.history[Number(button.dataset.index)])));
}

async function runBacktest(event) {
  event.preventDefault();
  const button = $("#run-backtest"); button.disabled = true; button.querySelector("span").textContent = btState.mode === "live" ? "正在下载历史行情" : "正在计算";
  const payload = {
    mode: btState.mode, start_date: $("#start-date").value, end_date: $("#end-date").value,
    codes: codesFromInput(), top_n: Number($("#top-n").value), rebalance_days: Number($("#rebalance-days").value),
    initial_capital: Number($("#initial-capital").value), cost_bps: Number($("#cost-bps").value),
    strategy: $("#strategy").value,
    rebalance_policy: $("#rebalance-policy").value,
    min_hold_days: Number($("#min-hold-days").value), rank_buffer: Number($("#rank-buffer").value),
    score_gap: Number($("#score-gap").value), risk_overlay: $("#risk-overlay").value,
    drawdown_limit: Number($("#drawdown-limit").value),
    drawdown_cooldown_days: Number($("#drawdown-cooldown").value),
  };
  try {
    const result = await request("/api/backtests/run", { method: "POST", body: JSON.stringify(payload) });
    renderResult(result);
    const data = await request("/api/backtests"); renderHistory(data.history);
    toast(`回测完成：${result.data.trading_days} 个交易日，${result.metrics.rebalance_count} 次调仓`);
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; button.querySelector("span").textContent = "开始回测"; }
}

document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) window.lucide.createIcons();
  const today = new Date(), start = new Date(today); start.setFullYear(today.getFullYear() - 2);
  $("#end-date").value = today.toISOString().slice(0, 10); $("#start-date").value = start.toISOString().slice(0, 10);
  document.querySelectorAll(".segment").forEach(button => button.addEventListener("click", () => {
    btState.mode = button.dataset.mode; document.querySelectorAll(".segment").forEach(item => item.classList.toggle("active", item === button));
  }));
  $("#stock-codes").addEventListener("input", updateUniverseCount); $("#backtest-form").addEventListener("submit", runBacktest);
  $("#rebalance-policy").addEventListener("change", updateAdaptiveFields); updateAdaptiveFields();
  window.addEventListener("resize", () => btState.chart?.resize());
  try {
    const data = await request("/api/backtests");
    $("#stock-codes").value = data.default_codes.join(", "); updateUniverseCount(); renderHistory(data.history); renderResult(data.latest);
  } catch (error) { toast(error.message); }
});
