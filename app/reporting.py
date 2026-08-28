import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from app.models import Candidate, MarketIndex, Sector, StrategyPick


class ReportGenerator:
    def __init__(self, report_dir: Path, api_key: str = "", base_url: str = "", model: str = ""):
        self.report_dir = report_dir
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        report_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        created_at: datetime,
        indices: List[MarketIndex],
        sectors: List[Sector],
        candidates: List[Candidate],
        breadth: Dict,
        strategy_picks: Optional[List[StrategyPick]] = None,
        strategy_signal: Optional[Dict] = None,
        macro_environment: Optional[Dict] = None,
    ) -> Dict[str, str]:
        deterministic = self._deterministic(
            created_at, indices, sectors, candidates, breadth, strategy_picks or [], strategy_signal or {},
            macro_environment or {},
        )
        markdown = deterministic["markdown"]
        source = "规则模板"
        warning: Optional[str] = None
        if self.api_key and candidates:
            try:
                markdown = self._enrich_with_llm(markdown, candidates, sectors, breadth, macro_environment or {})
                source = f"{self.model} + 结构化因子"
            except Exception as exc:
                warning = f"AI研报生成失败，已使用规则模板: {exc}"
        path = self.report_dir / f"{created_at.strftime('%Y%m%d_%H%M%S')}_research.md"
        path.write_text(markdown, encoding="utf-8")
        return {
            "markdown": markdown,
            "summary": deterministic["summary"],
            "source": source,
            "path": str(path),
            "warning": warning or "",
        }

    def _deterministic(
        self, created_at, indices, sectors, candidates, breadth, strategy_picks, strategy_signal,
        macro_environment,
    ):
        top_industries = [item for item in sectors if item.kind == "行业"][:5]
        top_concepts = [item for item in sectors if item.kind == "概念"][:5]
        total = max(1, breadth.get("up", 0) + breadth.get("down", 0) + breadth.get("flat", 0))
        up_ratio = breadth.get("up", 0) / total * 100
        mood = "偏强" if up_ratio >= 60 else "均衡" if up_ratio >= 45 else "偏弱"
        index_line = "，".join(f"{item.name} {item.pct_change:+.2f}%" for item in indices) or "指数数据暂缺"
        sector_line = "，".join(f"{item.name} {item.pct_change:+.2f}%" for item in top_industries) or "行业数据暂缺"
        summary = (
            f"市场{mood}，上涨 {breadth.get('up', 0)} 家、下跌 {breadth.get('down', 0)} 家。"
            f"行业前列：{sector_line}。今日量化候选 {len(candidates)} 只，"
            f"其中科创/科技专项 {sum(item.is_technology for item in candidates)} 只。"
        )
        lines = [
            f"# A股每日量化研究报告｜{created_at.strftime('%Y-%m-%d')}",
            "",
            "> 本报告由公开行情和透明因子规则生成，仅用于研究，不构成投资建议。",
            "",
            "## 市场概览",
            "",
            f"- 指数：{index_line}",
            f"- 市场宽度：上涨 {breadth.get('up', 0)} / 下跌 {breadth.get('down', 0)} / 平盘 {breadth.get('flat', 0)}",
            f"- 成交额：{breadth.get('amount', 0) / 1e8:.1f} 亿元",
            f"- 研判：市场状态{mood}。",
        ]
        if macro_environment:
            dimensions = macro_environment.get("dimensions", {})
            dimension_names = {
                "international": "国际市场",
                "domestic_policy": "国内政策",
                "sentiment": "舆论情绪",
            }
            lines.extend([
                "",
                "## 宏观环境评测",
                "",
                f"- 综合状态：**{macro_environment.get('regime', '中性')}**，{macro_environment.get('score', 50):.1f} 分",
                f"- 数据置信度：{macro_environment.get('confidence', 0):.0%}",
                f"- 研究建议总仓位：{macro_environment.get('recommended_exposure', 1.0):.0%}",
                f"- 基础策略保护：优先执行基础策略，宏观提示最多减仓 {macro_environment.get('max_macro_reduction', 0):.0%}",
                "- 执行状态：仅供风险提示，不自动改写交易订单",
            ])
            for key, label in dimension_names.items():
                dimension = dimensions.get(key, {})
                signals = "；".join(dimension.get("signals", [])[:3]) or "数据不足，按中性处理"
                lines.append(
                    f"- {label}：{dimension.get('score', 50):.1f} 分 / {dimension.get('label', '中性')}；{signals}"
                )
            for warning in macro_environment.get("warnings", [])[:3]:
                lines.append(f"- 宏观数据提示：{warning}")
        lines.extend([
            "",
            "## 板块强弱",
            "",
            "### 行业",
            "",
        ])
        lines.extend(
            f"- {item.name}：{item.pct_change:+.2f}%，上涨/下跌 {item.up_count}/{item.down_count}，领涨 {item.leading_stock or '暂无'}"
            for item in top_industries
        )
        lines.extend(["", "### 概念", ""])
        lines.extend(f"- {item.name}：{item.pct_change:+.2f}%" for item in top_concepts)
        decision = strategy_signal.get("decision") or {}
        paper = strategy_signal.get("paper_account") or {}
        if paper:
            lines.extend([
                "",
                "## LGBM 模拟盘账户",
                "",
                f"- 账户净值：{paper.get('nav', 0):,.2f}",
                f"- 当日盈亏：{paper.get('daily_pnl', 0):+,.2f}（{paper.get('daily_return', 0):+.2%}）",
                f"- 累计盈亏：{paper.get('cumulative_pnl', 0):+,.2f}（{paper.get('cumulative_return', 0):+.2%}）",
                f"- 当前回撤：{paper.get('drawdown', 0):.2%}",
                f"- 现金 / 仓位：{paper.get('cash', 0):,.2f} / {paper.get('exposure', 0):.1%}",
                f"- 累计交易成本：{paper.get('transaction_cost_total', 0):,.2f}",
                f"- 模型版本：`{strategy_signal.get('model_version', '--')}`",
                "- 交易口径：收盘产生信号，下一交易日开盘成交；买入按100股整数单位；单边成本12bp；不连接券商。",
            ])
            trades = paper.get("trades") or []
            lines.extend(["", "### 今日模拟成交", ""])
            if trades:
                lines.extend(
                    f"- {'买入' if item['side'] == 'BUY' else '卖出'} {item['name']}（{item['code']}）{item['shares']}股，成交价 {item['price']:.2f}，成本 {item['fee']:.2f}"
                    for item in trades
                )
            else:
                lines.append("- 无成交")
            positions = paper.get("positions") or []
            lines.extend(["", "### 当前模拟持仓", ""])
            if positions:
                lines.extend(
                    f"- {item['name']}（{item['code']}）：{item['shares']}股，收盘价 {item['last_price']:.2f}，市值 {item['market_value']:,.2f}，权重 {item['weight']:.1%}，浮盈亏 {item['unrealized_pnl']:+,.2f}"
                    for item in positions
                )
            else:
                lines.append("- 尚未建仓，等待上一收盘信号在下一交易日开盘执行。")
            for warning in paper.get("warnings", []):
                lines.append(f"- 模拟盘提示：{warning}")
        if decision:
            lines.extend([
                "",
                "## 每日自适应交易决策",
                "",
                f"> 今日动作：**{decision.get('action_label', '继续持有')}**；执行：{decision.get('execution_date', '无交易')}。",
                "",
                f"- 原因：{decision.get('reason', '--')}",
                f"- 周期：第 {decision.get('cycle_day', 0)} / {decision.get('rules', {}).get('rebalance_days', 10)} 个交易日，距下次主调仓 {decision.get('next_scheduled_in', '--')} 个交易日",
                f"- 提前换股额度：本周期已使用 {decision.get('adaptive_used_in_cycle', 0)} / 1 次",
            ])
            executed = decision.get("executed_order") or {}
            if executed:
                lines.append(
                    f"- 今日开盘执行：{executed.get('action_label', '--')}，对应 {executed.get('signal_date', '--')} 收盘信号"
                )
            if decision.get("trade_required"):
                sells = "、".join(f"{item['name']}（{item['code']}）" for item in decision.get("sells", [])) or "无"
                buys = "、".join(f"{item['name']}（{item['code']}）" for item in decision.get("buys", [])) or "无新增成分，仅恢复等权"
                lines.extend([f"- 下一交易日卖出：{sells}", f"- 下一交易日买入：{buys}"])
            else:
                lines.append("- 交易指令：无，继续持有")
            portfolio = decision.get("target_holdings") or decision.get("current_holdings") or []
            if portfolio:
                lines.append("- 目标持仓：" + "、".join(
                    f"{item['name']} {item.get('target_weight', 0):.0%}" for item in portfolio
                ))
        if strategy_picks:
            lines.extend([
                "",
                f"## 今日{strategy_signal.get('strategy', '量化策略')}观察 Top5",
                "",
                f"> 信号日 {strategy_signal.get('signal_date', '--')}。该排名用于每日自适应判断，不代表每天强制换仓。",
                "",
            ])
            lines.extend(
                f"- {item.rank}. {item.name}（{item.code}）：{item.score:.1f} 分，参考价 {item.price:.2f}，目标权重 {item.target_weight:.0%}；优势因子：{item.reason}"
                for item in strategy_picks
            )
        lines.extend(["", "## Top 候选", ""])
        if not candidates:
            lines.append("当前筛选条件下没有合格候选。")
        for item in candidates:
            factors = " / ".join(f"{name} {score:.0f}" for name, score in item.factor_scores.items())
            lines.extend([
                f"### {item.rank}. {item.name}（{item.code}）｜{item.total_score:.1f} 分",
                "",
                f"- 行情：{item.price:.2f} 元，今日 {item.pct_change:+.2f}%",
                f"- 板块：{item.board or '未分类'} / {item.industry}{(' / ' + item.concept) if item.concept else ''}{' / 科技专项' if item.is_technology else ''}",
                f"- 因子：{factors}",
                f"- 关注：{'；'.join(item.reasons)}",
                f"- 风险：{'；'.join(item.risks)}",
                f"- 数据置信度：{item.confidence:.0f}%",
                "",
            ])
        lines.extend([
            "## 次日检查清单",
            "",
            "- 候选股是否继续获得所属板块支持。",
            "- 开盘量价是否确认，而非单纯高开回落。",
            "- 市场宽度和总成交额是否同步改善。",
            "- 出现重大公告、监管或业绩事件时重新评估。",
        ])
        return {"markdown": "\n".join(lines), "summary": summary}

    def _enrich_with_llm(self, baseline, candidates, sectors, breadth, macro_environment):
        prompt = {
            "market_breadth": breadth,
            "top_sectors": [item.__dict__ for item in sectors[:10]],
            "macro_environment": macro_environment,
            "candidates": [item.__dict__ for item in candidates],
            "baseline_report": baseline,
        }
        system = (
            "你是严谨的A股量化研究员。只依据输入数据写中文收盘研报，不编造新闻、财务数据或目标价。"
            "必须区分事实、模型评分和风险；保留候选排名；用Markdown；明确声明不构成投资建议。"
        )
        with httpx.Client(timeout=90) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                    ],
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
