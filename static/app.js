const state = { mode: "demo", latest: null, history: [], backtests: [], activeBacktest: null, backtestChart: null, strategyDecision: null, candidateFilter: "all", candidates: [] };

const $ = (selector) => document.querySelector(selector);
const fmt = (value, digits = 2) => Number(value || 0).toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const signedClass = (value) => Number(value) >= 0 ? "positive" : "negative";
const percent = value => `${Number(value || 0) >= 0 ? "+" : ""}${(Number(value || 0) * 100).toFixed(2)}%`;

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(window.toastTimer);
  window.toastTimer = setTimeout(() => node.classList.remove("show"), 3200);
}

async function request(url, options = {}) {
  const response = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `请求失败 (${response.status})`);
  return data;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}

function markdown(markdownText) {
  const lines = String(markdownText || "").split("\n");
  let html = "", inList = false;
  const inline = text => escapeHtml(text).replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/`(.*?)`/g, "<code>$1</code>");
  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith("- ")) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${inline(line.slice(2))}</li>`;
      continue;
    }
    if (inList) { html += "</ul>"; inList = false; }
    if (line.startsWith("### ")) html += `<h3>${inline(line.slice(4))}</h3>`;
    else if (line.startsWith("## ")) html += `<h2>${inline(line.slice(3))}</h2>`;
    else if (line.startsWith("# ")) html += `<h1>${inline(line.slice(2))}</h1>`;
    else if (line.startsWith("> ")) html += `<blockquote>${inline(line.slice(2))}</blockquote>`;
    else if (line) html += `<p>${inline(line)}</p>`;
  }
  if (inList) html += "</ul>";
  return html;
}

function renderSectors(items, kind, target) {
  const sectors = items.filter(item => item.kind === kind).slice(0, 8);
  const max = Math.max(...sectors.map(item => Math.abs(item.pct_change)), 1);
  $(target).innerHTML = sectors.map(item => `
    <div class="sector-row ${Number(item.pct_change) < 0 ? "negative" : ""}">
      <span class="sector-name" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span>
      <span class="sector-bar"><i style="width:${Math.max(4, Math.abs(item.pct_change) / max * 100)}%"></i></span>
      <span class="sector-change ${signedClass(item.pct_change)}">${Number(item.pct_change) >= 0 ? "+" : ""}${fmt(item.pct_change)}%</span>
    </div>`).join("") || '<div class="sector-row"><span>暂无数据</span></div>';
}

function factorRows(scores) {
  const labels = { momentum: "趋势", sector: "板块", liquidity: "流动", value: "估值", risk: "风险" };
  return Object.entries(scores).map(([key, value]) => `
    <div class="factor-line"><span>${labels[key] || key}</span><span class="factor-track"><i style="width:${value}%"></i></span><b>${Math.round(value)}</b></div>`).join("");
}

function renderCandidates(items) {
  state.candidates = items;
  const visible = state.candidateFilter === "technology" ? items.filter(item => item.is_technology) : items;
  $("#candidate-body").innerHTML = visible.map(item => `
    <tr>
      <td><span class="rank">${String(item.rank).padStart(2, "0")}</span></td>
      <td><span class="stock-name">${escapeHtml(item.name)}</span><span class="stock-code">${escapeHtml(item.code)}</span></td>
      <td><span>${escapeHtml(item.board || item.industry)}${item.is_technology ? ' · <b class="tech-label">科创/科技</b>' : ""}</span><span class="stock-code">${escapeHtml(item.industry)}${item.concept ? ` / ${escapeHtml(item.concept)}` : ""}</span></td>
      <td><span class="price">${fmt(item.price)}</span><span class="change ${signedClass(item.pct_change)}">${Number(item.pct_change) >= 0 ? "+" : ""}${fmt(item.pct_change)}%</span></td>
      <td><span class="score">${fmt(item.total_score, 1)}</span><span class="stock-code">置信 ${fmt(item.confidence, 0)}%</span></td>
      <td class="factor-cell">${factorRows(item.factor_scores)}</td>
      <td><ul class="signal-list">${item.reasons.slice(0, 3).map(reason => `<li>${escapeHtml(reason)}</li>`).join("")}</ul></td>
    </tr>`).join("") || '<tr><td colspan="7">当前条件下没有候选股票</td></tr>';
  document.querySelectorAll(".candidate-tab").forEach(button => {
    const filter = button.dataset.candidateFilter;
    const count = filter === "technology" ? items.filter(item => item.is_technology).length : items.length;
    button.textContent = `${filter === "technology" ? "科创/科技" : "全部"}${count}`;
    button.classList.toggle("active", filter === state.candidateFilter);
  });
}

function strategyRows(items, emptyText) {
  return (items || []).map(item => `
    <div class="adaptive-holding-row">
      <span><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.code)} · 排名 ${item.rank ?? "--"}</small></span>
      <span><strong>${fmt(item.score, 1)}</strong><small>${item.holding_days == null ? escapeHtml(item.reason || "因子排名") : `已持有 ${item.holding_days} 天`}</small></span>
      <b>${Number(item.target_weight || 0).toLocaleString("zh-CN", { style: "percent", maximumFractionDigits: 0 })}</b>
    </div>`).join("") || `<p class="subtle">${escapeHtml(emptyText)}</p>`;
}

function paperRows(items, emptyText, mode = "position") {
  return (items || []).map(item => {
    const weight = Number(item.weight ?? item.target_weight ?? 0);
    const secondary = mode === "position"
      ? `${Number(item.shares || 0).toLocaleString("zh-CN")} 股 · 市值 ${fmt(item.market_value, 2)}`
      : `目标 ${(weight * 100).toFixed(0)}% · 信号价 ${fmt(item.price, 2)}`;
    const value = mode === "position" ? `${(weight * 100).toFixed(1)}%` : `排名 ${item.rank ?? "--"}`;
    return `
      <div class="paper-list-row">
        <span><strong>${escapeHtml(item.name || item.code)}</strong><small>${escapeHtml(item.code)}</small></span>
        <span><strong>${secondary}</strong><small>${escapeHtml(item.reason || "")}</small></span>
        <b>${value}</b>
      </div>`;
  }).join("") || `<p class="subtle">${escapeHtml(emptyText)}</p>`;
}

function renderPaperAccount(snapshot, accountState, portfolioState, modelStatus = {}) {
  const empty = $("#paper-account-empty");
  const content = $("#paper-account-content");
  if (!snapshot) {
    empty.hidden = false;
    content.hidden = true;
    return;
  }
  empty.hidden = true;
  content.hidden = false;
  const decision = portfolioState?.last_decision || {};
  const pending = portfolioState?.pending_order || decision.pending_order || null;
  const positions = snapshot.positions || [];
  const trades = snapshot.trades || [];
  const waiting = Boolean(pending) && positions.length === 0;

  $("#paper-account-status").textContent = waiting ? "等待下一交易日开盘建仓" : (positions.length ? "模拟持仓运行中" : "账户已建立");
  $("#paper-account-id").textContent = snapshot.account_id || accountState?.account_id || "--";
  $("#paper-model-version").textContent = modelStatus.model_version || "--";
  $("#paper-signal-date").textContent = snapshot.signal_date || "--";
  $("#paper-nav").textContent = `¥${fmt(snapshot.nav, 2)}`;
  $("#paper-daily-pnl").textContent = `${Number(snapshot.daily_pnl || 0) >= 0 ? "+" : ""}¥${fmt(snapshot.daily_pnl, 2)}`;
  $("#paper-daily-pnl").className = signedClass(snapshot.daily_pnl);
  $("#paper-cumulative-return").textContent = percent(snapshot.cumulative_return);
  $("#paper-cumulative-return").className = signedClass(snapshot.cumulative_return);
  $("#paper-drawdown").textContent = percent(snapshot.drawdown);
  $("#paper-drawdown").className = Number(snapshot.drawdown || 0) < 0 ? "negative" : "";
  $("#paper-cash").textContent = `¥${fmt(snapshot.cash, 2)}`;
  $("#paper-exposure").textContent = `${(Number(snapshot.exposure || 0) * 100).toFixed(1)}%`;
  $("#paper-cost").textContent = `¥${fmt(snapshot.transaction_cost_total, 2)}`;
  $("#paper-execution-status").textContent = pending ? `计划：${pending.execution_date || "下一交易日开盘"}` : "当前没有待执行订单";
  $("#paper-execution-detail").textContent = pending
    ? `${pending.action_label || "调仓"} · ${pending.reason || "按策略信号执行"} · 单边成本 ${fmt(snapshot.execution?.cost_bps_one_way, 0)}bp`
    : `收盘信号 · 下一交易日开盘成交 · 单边成本 ${fmt(snapshot.execution?.cost_bps_one_way, 0)}bp`;
  $("#paper-positions").innerHTML = paperRows(positions, "等待首笔订单在下一交易日开盘成交", "position");
  $("#paper-order-title").textContent = pending ? "待执行订单" : "最新成交";
  $("#paper-pending-order").innerHTML = pending
    ? paperRows(pending.target_holdings || pending.buys, "待执行订单为空", "pending")
    : paperRows(trades, "当前没有待执行订单或当日成交", "trade");
  const tradeNode = $("#paper-trades");
  tradeNode.hidden = !trades.length;
  tradeNode.textContent = trades.length
    ? `当日成交：${trades.map(item => `${item.side === "BUY" ? "买入" : "卖出"}${item.name || item.code} ${Number(item.shares || 0).toLocaleString("zh-CN")}股 @ ${fmt(item.price, 2)}`).join("；")}`
    : "";
}

function renderAdaptiveDecision(decision, watchlist = []) {
  if (!decision) return;
  state.strategyDecision = decision;
  const fixedCycle = decision.rules?.adaptive_enabled === false;
  $("#strategy-section-title").textContent = fixedCycle ? "固定 10 日调仓 Top5" : "每日自适应 Top5";
  $("#strategy-policy-tag").textContent = fixedCycle
    ? "固定10日主调仓 · 周期内不提前换股 · 次日开盘执行"
    : "10日主调仓 · 强信号最多提前换1只 · 次日开盘执行";
  $("#adaptive-empty").hidden = true;
  $("#adaptive-content").hidden = false;
  $("#adaptive-action").textContent = decision.action_label || "继续持有";
  $("#adaptive-action").className = decision.trade_required ? (decision.status === "risk" ? "negative" : "positive") : "";
  $("#adaptive-date").textContent = decision.signal_date || "--";
  $("#adaptive-cycle").textContent = `${decision.cycle_day ?? 0} / ${decision.rules?.rebalance_days ?? 10} 日`;
  $("#adaptive-next").textContent = `${decision.next_scheduled_in ?? "--"} 个交易日`;
  $("#adaptive-execution").textContent = decision.trade_required ? `计划：${decision.execution_date}` : "今日不交易";
  $("#adaptive-reason").textContent = decision.reason || "--";
  $("#adaptive-quota").hidden = fixedCycle;
  $("#adaptive-quota").textContent = fixedCycle ? "" : `提前换股 ${decision.adaptive_used_in_cycle ?? 0} / 1`;
  const orders = $("#adaptive-orders");
  const buyText = (decision.buys || []).map(item => `${escapeHtml(item.name)}(${escapeHtml(item.code)})`).join("、");
  const sellText = (decision.sells || []).map(item => `${escapeHtml(item.name)}(${escapeHtml(item.code)})`).join("、");
  orders.hidden = !decision.trade_required;
  orders.innerHTML = decision.trade_required ? `
    <span><b>卖出</b>${sellText || "无"}</span>
    <span><b>买入</b>${buyText || "无新增成分，仅恢复等权"}</span>` : "";
  const portfolio = decision.target_holdings?.length ? decision.target_holdings : decision.current_holdings;
  $("#adaptive-portfolio-title").textContent = decision.trade_required ? "下一交易日目标持仓" : "当前策略持仓";
  $("#adaptive-holdings").innerHTML = strategyRows(portfolio, "等待首次建仓执行");
  $("#adaptive-watchlist").innerHTML = strategyRows(watchlist, "今日观察榜随实时选股生成");
}

function renderRun(run) {
  if (!run) return;
  state.latest = run;
  state.mode = run.mode;
  document.querySelectorAll(".segment").forEach(button => button.classList.toggle("active", button.dataset.mode === state.mode));
  $("#market-state span:last-child").textContent = run.market_status;
  $("#metric-up").textContent = run.market_breadth.up ?? "--";
  $("#metric-down").textContent = run.market_breadth.down ?? "--";
  $("#metric-amount").textContent = fmt((run.market_breadth.amount || 0) / 1e8, 1);
  $("#metric-candidates").textContent = run.candidates.length;
  $("#updated-at").textContent = new Date(run.created_at).toLocaleString("zh-CN", { hour12: false });
  $("#market-summary").textContent = run.report_summary;
  const warningStrip = $("#warning-strip");
  warningStrip.hidden = !run.warnings?.length;
  warningStrip.textContent = run.warnings?.map(item => item.split(":")[0]).join("；") || "";
  $("#index-strip").innerHTML = run.indices.map(item => `
    <div class="index-item"><span>${escapeHtml(item.name)}</span><strong>${fmt(item.price)} <small class="${signedClass(item.pct_change)}">${Number(item.pct_change) >= 0 ? "+" : ""}${fmt(item.pct_change)}%</small></strong></div>`).join("") || '<div class="index-item"><span>指数数据暂缺</span></div>';
  renderSectors(run.sectors, "行业", "#industry-list");
  renderSectors(run.sectors, "概念", "#concept-list");
  renderCandidates(run.candidates);
  const decision = run.strategy_signal?.decision;
  if (decision) renderAdaptiveDecision(decision, run.strategy_picks || []);
  $("#report-content").innerHTML = markdown(run.report_markdown);
  if (window.lucide) window.lucide.createIcons();
}

function renderHistory(items) {
  state.history = items;
  $("#history-list").innerHTML = items.map((item, index) => `
    <button class="history-item" data-index="${index}"><span>${new Date(item.created_at).toLocaleDateString("zh-CN")}</span><small>${item.mode === "live" ? "实时A股" : "演示数据"} · ${item.candidates.length} 只候选</small></button>`).join("");
  document.querySelectorAll(".history-item").forEach(button => button.addEventListener("click", () => renderRun(state.history[Number(button.dataset.index)])));
}

function renderEmbeddedBacktestChart(result) {
  if (!window.echarts || !result?.curves?.length) return;
  if (!state.backtestChart) state.backtestChart = echarts.init($("#selection-backtest-chart"));
  const dates = result.curves.map(item => item.date);
  const compact = window.innerWidth <= 520;
  state.backtestChart.setOption({
    animationDuration: 450,
    textStyle: { fontFamily: '"Noto Sans SC", sans-serif', color: "#69716d" },
    tooltip: {
      trigger: "axis",
      formatter: params => {
        const rows = params.map(item => `${item.marker}${item.seriesName}：${item.seriesName === "回撤" ? percent(item.value) : fmt(item.value, 4)}`);
        return `<strong>${params[0]?.axisValue || ""}</strong><br>${rows.join("<br>")}`;
      },
    },
    legend: { top: 10, right: compact ? 0 : 8, itemWidth: compact ? 18 : 25, textStyle: { fontSize: compact ? 10 : 12 }, data: ["策略净值", "等权基准"] },
    grid: [
      { left: compact ? 44 : 52, right: compact ? 8 : 22, top: 50, height: compact ? "50%" : "55%" },
      { left: compact ? 44 : 52, right: compact ? 8 : 22, top: compact ? "70%" : "74%", height: compact ? "20%" : "14%" },
    ],
    xAxis: [
      { type: "category", data: dates, boundaryGap: false, axisLabel: { show: !compact, hideOverlap: true }, axisLine: { lineStyle: { color: "#dfe3e0" } } },
      { type: "category", data: dates, boundaryGap: false, gridIndex: 1, axisLabel: { hideOverlap: true, interval: compact ? Math.max(0, Math.ceil(dates.length / 4) - 1) : "auto", fontSize: compact ? 9 : 12, formatter: value => compact ? value.slice(0, 7) : value }, axisLine: { lineStyle: { color: "#dfe3e0" } } },
    ],
    yAxis: [
      { type: "value", scale: true, splitNumber: compact ? 4 : 5, axisLabel: { fontSize: compact ? 9 : 12 }, splitLine: { lineStyle: { color: "#e3e7e4" } } },
      { type: "value", gridIndex: 1, splitNumber: compact ? 3 : 5, axisLabel: { fontSize: compact ? 9 : 12, formatter: value => `${(value * 100).toFixed(0)}%` }, splitLine: { lineStyle: { color: "#e3e7e4" } } },
    ],
    dataZoom: [{ type: "inside", xAxisIndex: [0, 1] }],
    series: [
      { name: "策略净值", type: "line", data: result.curves.map(item => item.strategy), showSymbol: false, lineStyle: { width: 2.2, color: "#16724a" }, itemStyle: { color: "#16724a" }, xAxisIndex: 0, yAxisIndex: 0 },
      { name: "等权基准", type: "line", data: result.curves.map(item => item.benchmark), showSymbol: false, lineStyle: { width: 1.6, color: "#a97818" }, itemStyle: { color: "#a97818" }, xAxisIndex: 0, yAxisIndex: 0 },
      { name: "回撤", type: "line", data: result.curves.map(item => item.drawdown), showSymbol: false, lineStyle: { width: 1, color: "#b8453c" }, areaStyle: { color: "rgba(184,69,60,.16)" }, xAxisIndex: 1, yAxisIndex: 1 },
    ],
  }, true);
  requestAnimationFrame(() => state.backtestChart?.resize());
}

function renderEmbeddedBacktest(result) {
  const empty = $("#embedded-backtest-empty");
  const content = $("#embedded-backtest-content");
  if (!result) {
    empty.hidden = false;
    content.hidden = true;
    return;
  }
  empty.hidden = true;
  content.hidden = false;
  state.activeBacktest = result;
  const metrics = result.metrics;
  const isLive = result.config.mode === "live";
  const excessPositive = Number(metrics.excess_return) >= 0;
  $("#embedded-bt-title").textContent = `${result.data.strategy || "原始多因子"} · ${result.data.policy_label || "固定周期"} · ${isLive ? "真实历史" : "演示数据"} · Top ${result.config.top_n}`;
  $("#embedded-bt-period").textContent = `${result.config.start_date} 至 ${result.config.end_date} · ${result.data.trading_days} 个交易日 · ${result.config.codes.length} 只股票`;
  $("#embedded-bt-verdict").textContent = excessPositive ? "跑赢等权基准" : "未跑赢等权基准";
  $("#embedded-bt-verdict").className = `performance-verdict ${excessPositive ? "positive" : "negative"}`;
  const values = [
    ["#embedded-bt-total", metrics.total_return], ["#embedded-bt-annual", metrics.annual_return],
    ["#embedded-bt-benchmark", metrics.benchmark_return], ["#embedded-bt-excess", metrics.excess_return],
  ];
  values.forEach(([selector, value]) => {
    $(selector).textContent = percent(value);
    $(selector).className = signedClass(value);
  });
  $("#embedded-bt-drawdown").textContent = percent(metrics.max_drawdown);
  $("#embedded-bt-drawdown").className = "negative";
  $("#embedded-bt-risk").textContent = `${fmt(metrics.sharpe)} / ${(Number(metrics.average_turnover || 0) * 100).toFixed(2)}%`;
  const warning = $("#embedded-bt-warnings");
  warning.hidden = !result.warnings?.length;
  warning.textContent = result.warnings?.join("；") || "";
  document.querySelectorAll(".embedded-history-item").forEach(button => button.classList.toggle("active", button.dataset.id === result.backtest_id));
  renderEmbeddedBacktestChart(result);
}

function renderEmbeddedBacktestHistory(items) {
  state.backtests = items;
  $("#embedded-backtest-history").innerHTML = items.slice(0, 4).map(item => `
    <button class="embedded-history-item" data-id="${escapeHtml(item.backtest_id)}">
      <strong>${escapeHtml(item.config.start_date)} → ${escapeHtml(item.config.end_date)}</strong>
      <span>${item.data.policy_label || "固定周期"} · ${item.config.mode === "live" ? "真实" : "演示"} · ${percent(item.metrics.total_return)} · 超额 ${percent(item.metrics.excess_return)}</span>
    </button>`).join("");
  document.querySelectorAll(".embedded-history-item").forEach(button => button.addEventListener("click", () => {
    const result = state.backtests.find(item => item.backtest_id === button.dataset.id);
    renderEmbeddedBacktest(result);
  }));
}

function renderBacktestOverview(data) {
  renderEmbeddedBacktestHistory(data?.history || []);
  renderEmbeddedBacktest(data?.latest || null);
}

async function loadDashboard() {
  const [dashboard, history, backtests] = await Promise.all([
    request("/api/dashboard"),
    request("/api/history?limit=12"),
    request("/api/backtests?limit=4").catch(() => null),
  ]);
  const persistedDecision = dashboard.strategy?.state?.last_decision || null;
  if (persistedDecision) renderAdaptiveDecision(persistedDecision, dashboard.latest?.strategy_picks || []);
  renderRun(dashboard.latest);
  renderPaperAccount(
    dashboard.strategy?.paper_account,
    dashboard.strategy?.paper_account_state,
    dashboard.strategy?.state,
    dashboard.settings?.lgbm_model_status || {},
  );
  renderHistory(history.items);
  renderBacktestOverview(backtests);
  const config = dashboard.settings;
  $("#metric-next-run").textContent = config.next_run ? new Date(config.next_run).toLocaleString("zh-CN", { weekday: "short", hour: "2-digit", minute: "2-digit" }) : config.schedule_time;
  $("#metric-channels").textContent = config.channels.length ? config.channels.join(" · ") : "未配置推送渠道";
  const lgbm = config.lgbm_research;
  const backendLabels = { contrarian: "当前：规则策略", lgbm_shadow: "当前：LGBM影子", lgbm_active: "当前：LGBM模拟盘" };
  const modelTag = $("#strategy-model-tag");
  if (modelTag) {
    const suffix = lgbm?.available ? ` · ${lgbm.model_version || "LGBM已冻结"}` : " · LGBM模型不可用";
    modelTag.textContent = `${backendLabels[config.strategy_model_backend] || "当前：规则策略"}${suffix}`;
  }
}

async function runAnalysis() {
  const button = $("#run-button");
  button.disabled = true;
  button.querySelector("span").textContent = state.mode === "live" ? "正在获取实时行情" : "正在计算";
  try {
    const result = await request("/api/run", { method: "POST", body: JSON.stringify({ mode: state.mode, notify: false }) });
    renderRun(result);
    renderPaperAccount(
      result.strategy_signal?.paper_account,
      null,
      { pending_order: result.strategy_signal?.decision?.pending_order, last_decision: result.strategy_signal?.decision },
      { model_version: result.strategy_signal?.model_version },
    );
    const history = await request("/api/history?limit=12");
    renderHistory(history.items);
    toast(`选股完成，生成 ${result.candidates.length} 只候选`);
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; button.querySelector("span").textContent = "运行选股"; }
}

async function pushReport() {
  const button = $("#push-button");
  button.disabled = true;
  try {
    const data = await request("/api/push", { method: "POST" });
    toast(data.results.map(item => `${item.channel}: ${item.detail}`).join("；"));
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}

document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) window.lucide.createIcons();
  document.querySelectorAll(".segment").forEach(button => button.addEventListener("click", () => {
    state.mode = button.dataset.mode;
    document.querySelectorAll(".segment").forEach(item => item.classList.toggle("active", item === button));
  }));
  $("#run-button").addEventListener("click", runAnalysis);
  $("#push-button").addEventListener("click", pushReport);
  document.querySelectorAll(".candidate-tab").forEach(button => button.addEventListener("click", () => {
    state.candidateFilter = button.dataset.candidateFilter;
    renderCandidates(state.candidates);
  }));
  $("#refresh-button").addEventListener("click", () => loadDashboard().then(() => toast("已刷新")));
  $("#copy-report").addEventListener("click", async () => {
    if (!state.latest) return;
    await navigator.clipboard.writeText(state.latest.report_markdown);
    toast("研报全文已复制");
  });
  window.addEventListener("resize", () => {
    state.backtestChart?.resize();
    if (state.activeBacktest) renderEmbeddedBacktestChart(state.activeBacktest);
  });
  try { await loadDashboard(); } catch (error) { toast(error.message); }
});
