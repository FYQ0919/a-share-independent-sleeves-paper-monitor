const miningState = { data: null, filter: "all", charts: [] };

const percent = (value, digits = 2) => value == null ? "--" : `${(Number(value) * 100).toFixed(digits)}%`;
const number = (value, digits = 3) => value == null ? "--" : Number(value).toFixed(digits);
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

const familyLabels = {
  valuation: "估值",
  price_trend: "价格趋势",
  risk_tail: "风险尾部",
  liquidity: "流动性",
  microstructure_proxy: "微观结构代理",
  generated: "自动生成",
  other: "其他",
};

const reasonLabels = {
  selected: "入选",
  discovery_days: "发现样本不足",
  selection_days: "选择样本不足",
  weak_discovery_ic: "发现 IC 偏弱",
  unstable_discovery_direction: "发现方向不稳",
  unstable_selection_direction: "选择方向不稳",
  selection_sign_reversal: "选择集反向",
  redundant: "高度相关去重",
  factor_limit: "超出数量上限",
};

function renderSummary(data) {
  const selection = data.portfolio_evaluation.selection.factor_ensemble.performance;
  document.getElementById("factor-count").textContent = data.feature_count;
  const alphaCount = data.alpha158_factor_count ?? data.base_factor_count ?? data.feature_count;
  const frameworkCount = data.framework_factor_count ?? 0;
  document.getElementById("factor-count-detail").textContent = `${alphaCount} Alpha158 + ${frameworkCount} 复现 + ${data.generated_factor_count ?? 0} 生成`;
  document.getElementById("selected-count").textContent = data.selected_factor_count;
  document.getElementById("selected-count-detail").textContent = `${data.selected_framework_factor_count ?? 0} 复现 + ${data.selected_generated_factor_count ?? 0} 生成`;
  document.getElementById("selection-annual").textContent = percent(selection.annual_return);
  document.getElementById("selection-sharpe").textContent = number(selection.sharpe);
  const selectionPassed = Boolean(data.candidate_gate.passed);
  const diagnosticPassed = Boolean(data.revealed_diagnostic_review?.passed);
  const gate = document.getElementById("gate-status");
  gate.textContent = selectionPassed && diagnosticPassed ? "选择与诊断通过 · 未部署" : selectionPassed ? "选择通过 · 诊断失败" : "选择门禁未通过";
  gate.className = selectionPassed && diagnosticPassed ? "positive" : "negative";
  document.getElementById("generated-at").textContent = new Date(data.generated_at).toLocaleString("zh-CN", { hour12: false });
  document.getElementById("registry-version").textContent = data.registry_version ? `注册表 ${data.registry_version}` : "固定因子库";
  document.getElementById("mining-status").textContent = selectionPassed && diagnosticPassed ? "候选通过，研究态" : "研究完成，禁止部署";
  document.getElementById("mining-status-dot").classList.add(selectionPassed && diagnosticPassed ? "passed" : "rejected");
  const config = data.config;
  document.getElementById("factor-window-label").textContent = `${config.discovery.start}—${config.selection.end} 选择`;
}

function chartTextStyle() {
  return { color: "#69716d", fontFamily: '"Noto Sans SC", sans-serif', fontSize: 11 };
}

function renderCharts(data) {
  miningState.charts.forEach((chart) => chart.dispose());
  const scatterElement = document.getElementById("ic-scatter");
  const familyElement = document.getElementById("family-chart");
  const scatter = echarts.init(scatterElement);
  const family = echarts.init(familyElement);
  miningState.charts = [scatter, family];
  const rows = data.factor_diagnostics;
  const points = rows.map((row) => ({
    name: row.feature,
    value: [row.discovery_mean_rank_ic, row.selection_mean_rank_ic, row.selected ? 1 : 0],
    family: row.family,
    reason: row.selection_reason,
    itemStyle: { color: row.selected ? "#16724a" : "#aeb6b1", opacity: row.selected ? 0.95 : 0.55 },
    symbolSize: row.selected ? 10 : 6,
  }));
  const extent = Math.max(0.03, ...points.flatMap((point) => point.value.slice(0, 2).map((value) => Math.abs(value || 0)))) * 1.12;
  scatter.setOption({
    animation: false,
    title: { text: "方向稳定性", left: 0, textStyle: { color: "#202523", fontSize: 14, fontWeight: 600 } },
    grid: { left: 58, right: 20, top: 48, bottom: 48 },
    tooltip: {
      trigger: "item",
      formatter: (params) => `${escapeHtml(params.data.name)}<br>发现 IC ${number(params.value[0], 4)}<br>选择 IC ${number(params.value[1], 4)}<br>${reasonLabels[params.data.reason] || params.data.reason}`,
    },
    xAxis: {
      name: "发现集 Rank IC",
      nameLocation: "middle",
      nameGap: 30,
      min: -extent,
      max: extent,
      axisLabel: { ...chartTextStyle(), formatter: (value) => Number(value).toFixed(2) },
      axisLine: { lineStyle: { color: "#aeb6b1" } },
      splitLine: { lineStyle: { color: "#edf0ee" } },
    },
    yAxis: {
      name: "选择集 Rank IC",
      nameLocation: "middle",
      nameGap: 42,
      min: -extent,
      max: extent,
      axisLabel: { ...chartTextStyle(), formatter: (value) => Number(value).toFixed(2) },
      axisLine: { show: true, lineStyle: { color: "#aeb6b1" } },
      splitLine: { lineStyle: { color: "#edf0ee" } },
    },
    series: [{ type: "scatter", data: points, emphasis: { scale: 1.4 } }],
  });

  const selected = rows.filter((row) => row.selected);
  const counts = Object.entries(selected.reduce((result, row) => {
    result[row.family] = (result[row.family] || 0) + 1;
    return result;
  }, {})).sort((left, right) => right[1] - left[1]);
  family.setOption({
    animation: false,
    title: { text: "入选因子结构", left: 0, textStyle: { color: "#202523", fontSize: 14, fontWeight: 600 } },
    grid: { left: 100, right: 30, top: 48, bottom: 32 },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (value) => `${value} 个` },
    xAxis: {
      type: "value",
      minInterval: 1,
      axisLabel: chartTextStyle(),
      splitLine: { lineStyle: { color: "#edf0ee" } },
    },
    yAxis: {
      type: "category",
      data: counts.map(([name]) => familyLabels[name] || name),
      axisLabel: chartTextStyle(),
      axisLine: { lineStyle: { color: "#aeb6b1" } },
      axisTick: { show: false },
    },
    series: [{
      type: "bar",
      data: counts.map(([, count]) => count),
      barMaxWidth: 22,
      itemStyle: { color: "#16724a" },
      label: { show: true, position: "right", color: "#202523", fontSize: 11 },
    }],
  });
}

function performanceItem(label, strategy, performance, activity) {
  return `<div class="performance-item">
    <span>${escapeHtml(label)} · ${escapeHtml(strategy)}</span>
    <strong>${percent(performance.annual_return)}</strong>
    <small>回撤 ${percent(performance.max_drawdown)} · 夏普 ${number(performance.sharpe)} · 换手 ${number(activity.annual_turnover, 2)}x</small>
  </div>`;
}

function renderPerformance(data) {
  const selection = data.portfolio_evaluation.selection;
  const diagnostic = data.portfolio_evaluation.revealed_diagnostic;
  const selectionConfig = data.config.selection;
  const diagnosticConfig = data.config.diagnostic;
  const selectionLabel = `${selectionConfig.start}—${selectionConfig.end} 选择`;
  const diagnosticLabel = `${diagnosticConfig.start}—${diagnosticConfig.end} 隔离`;
  const items = [
    [selectionLabel, "因子组合", selection.factor_ensemble],
    [selectionLabel, "规则基线", selection.current_rule_baseline],
    [diagnosticLabel, "因子组合", diagnostic.factor_ensemble],
    [diagnosticLabel, "规则基线", diagnostic.current_rule_baseline],
  ];
  document.getElementById("portfolio-performance").innerHTML = items.map(([label, strategy, row]) =>
    performanceItem(label, strategy, row.performance, row.activity)
  ).join("");
}

function renderFactorTable() {
  const rows = miningState.data.factor_diagnostics.filter((row) => {
    if (miningState.filter === "selected") return row.selected;
    if (miningState.filter === "rejected") return !row.selected;
    return true;
  });
  document.getElementById("factor-table-body").innerHTML = rows.map((row) => `<tr>
    <td><strong class="factor-name">${escapeHtml(row.feature)}</strong></td>
    <td>${escapeHtml(familyLabels[row.family] || row.family)}</td>
    <td class="mono-number ${row.discovery_mean_rank_ic >= 0 ? "positive-text" : "negative-text"}">${number(row.discovery_mean_rank_ic, 4)}</td>
    <td class="mono-number ${row.selection_mean_rank_ic * row.discovery_direction >= 0 ? "positive-text" : "negative-text"}">${number(row.selection_mean_rank_ic, 4)}</td>
    <td class="mono-number">${number(row.diagnostic_mean_rank_ic, 4)}</td>
    <td class="mono-number">${percent(row.selection_direction_hit_rate, 1)}</td>
    <td class="mono-number">${number(row.stability_score, 5)}</td>
    <td class="mono-number">${number(row.max_abs_corr_to_selected, 3)}</td>
    <td><span class="factor-verdict ${row.selected ? "selected" : ""}">${escapeHtml(reasonLabels[row.selection_reason] || row.selection_reason)}</span></td>
  </tr>`).join("");
}

function renderWarnings(data) {
  const element = document.getElementById("mining-warnings");
  if (!data.warnings?.length) return;
  element.hidden = false;
  element.textContent = data.warnings.join("；");
}

async function loadMiningResult() {
  try {
    const response = await fetch("/api/factor-mining/latest");
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "载入失败");
    miningState.data = data;
    renderSummary(data);
    renderCharts(data);
    renderPerformance(data);
    renderFactorTable();
    renderWarnings(data);
  } catch (error) {
    document.getElementById("mining-status").textContent = error.message;
    document.getElementById("gate-status").textContent = "暂无结果";
    document.getElementById("gate-status").className = "negative";
  }
}

async function updateFactors() {
  const button = document.getElementById("update-factors");
  const status = document.getElementById("mining-status");
  button.disabled = true;
  button.classList.add("is-spinning");
  status.textContent = "正在生成和验证新因子";
  try {
    const response = await fetch("/api/factor-mining/update", { method: "POST" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "自动更新失败");
    await loadMiningResult();
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
    button.classList.remove("is-spinning");
  }
}

document.querySelectorAll("[data-filter]").forEach((button) => {
  button.addEventListener("click", () => {
    miningState.filter = button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach((item) => item.classList.toggle("active", item === button));
    renderFactorTable();
  });
});

document.getElementById("update-factors").addEventListener("click", updateFactors);

window.addEventListener("resize", () => miningState.charts.forEach((chart) => chart.resize()));
window.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) window.lucide.createIcons();
  loadMiningResult();
});
