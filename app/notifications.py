import base64
from email.message import EmailMessage
import hashlib
from pathlib import Path
import smtplib
from typing import Dict, List

import httpx

from app.config import Settings
from app.models import Candidate, Sector, StrategyPick


class NotificationService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def send(
        self,
        title: str,
        summary: str,
        report: str,
        candidates: List[Candidate],
        sectors: List[Sector],
        strategy_picks: List[StrategyPick] = None,
        strategy_signal: Dict = None,
        macro_environment: Dict = None,
    ):
        message = self._compact_message(
            title, summary, candidates, sectors, strategy_picks or [], strategy_signal or {},
            macro_environment or {},
        )
        results: List[Dict] = []
        if self.settings.feishu_webhook_url:
            results.append(self._post(
                "飞书",
                self.settings.feishu_webhook_url,
                {"msg_type": "text", "content": {"text": message[:18000]}},
            ))
        if self.settings.wecom_webhook_url:
            results.append(self._post(
                "企业微信",
                self.settings.wecom_webhook_url,
                {"msgtype": "markdown", "markdown": {"content": message[:3800]}},
            ))
            image_path = (strategy_signal or {}).get("paper_curve", {}).get("image_path")
            if image_path:
                results.append(self._send_wecom_image(Path(image_path)))
        if self.settings.generic_webhook_url:
            results.append(self._post(
                "通用 Webhook",
                self.settings.generic_webhook_url,
                {"title": title, "summary": summary, "content": message, "report": report},
            ))
        if self.settings.smtp_host and self.settings.email_to:
            results.append(self._email(title, report))
        if not results:
            results.append({"channel": "未配置", "ok": False, "detail": "研报已保存，尚未配置推送渠道"})
        return results

    def _send_wecom_image(self, path: Path) -> Dict:
        try:
            payload = self._wecom_image_payload(path)
        except Exception as exc:
            return {"channel": "企业微信图片", "ok": False, "detail": str(exc)}
        return self._post(
            "企业微信图片",
            self.settings.wecom_webhook_url,
            payload,
        )

    @staticmethod
    def _wecom_image_payload(path: Path) -> Dict:
        if not path.is_file():
            raise FileNotFoundError(f"收益曲线图片不存在: {path}")
        content = path.read_bytes()
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("企业微信收益曲线必须是有效 PNG 文件")
        if len(content) > 2 * 1024 * 1024:
            raise ValueError("企业微信机器人图片不能超过 2MB")
        return {
            "msgtype": "image",
            "image": {
                "base64": base64.b64encode(content).decode("ascii"),
                "md5": hashlib.md5(content).hexdigest(),
            },
        }

    @staticmethod
    def _compact_message(
        title: str,
        summary: str,
        candidates: List[Candidate],
        sectors: List[Sector],
        strategy_picks: List[StrategyPick],
        strategy_signal: Dict,
        macro_environment: Dict = None,
    ) -> str:
        top_sectors = "、".join(f"{item.name} {item.pct_change:+.2f}%" for item in sectors[:5])
        lines = [f"## {title}", "", summary, "", f"强势板块：{top_sectors or '暂无'}"]
        macro_environment = macro_environment or {}
        if macro_environment:
            dimensions = macro_environment.get("dimensions", {})
            international = dimensions.get("international", {})
            policy = dimensions.get("domestic_policy", {})
            sentiment = dimensions.get("sentiment", {})
            lines.extend([
                "",
                "### 宏观环境评测",
                f"> **{macro_environment.get('regime', '中性')}｜{macro_environment.get('score', 50):.1f} 分**｜数据置信度 {macro_environment.get('confidence', 0):.0%}",
                f"> 国际 {international.get('score', 50):.1f}｜国内政策 {policy.get('score', 50):.1f}｜舆情 {sentiment.get('score', 50):.1f}",
                f"> 研究建议总仓位 {macro_environment.get('recommended_exposure', 1.0):.0%}；基础策略优先，宏观最多提示减仓 {macro_environment.get('max_macro_reduction', 0):.0%}；**尚未自动改写交易订单**。",
            ])
        paper = strategy_signal.get("paper_account") or {}
        if paper:
            lines.extend([
                "",
                "### LGBM 模拟盘",
                f"> **净值 {paper.get('nav', 0):,.2f}**｜当日 {paper.get('daily_pnl', 0):+,.2f}（{paper.get('daily_return', 0):+.2%}）｜累计 {paper.get('cumulative_return', 0):+.2%}",
                f"> 回撤 {paper.get('drawdown', 0):.2%}｜现金 {paper.get('cash', 0):,.2f}｜仓位 {paper.get('exposure', 0):.1%}｜累计成本 {paper.get('transaction_cost_total', 0):,.2f}",
                f"> 模型：{strategy_signal.get('model_version', '--')}；信号日：{paper.get('signal_date', '--')}；仅为模拟盘，不连接券商。",
            ])
            trades = paper.get("trades") or []
            if trades:
                lines.append("今日模拟成交：" + "、".join(
                    f"{'买入' if item['side'] == 'BUY' else '卖出'}{item['name']} {item['shares']}股@{item['price']:.2f}"
                    for item in trades
                ))
            else:
                lines.append("今日模拟成交：无。")
            positions = paper.get("positions") or []
            if positions:
                lines.append("模拟持仓：" + "、".join(
                    f"{item['name']} {item['shares']}股/{item.get('weight', 0):.1%}"
                    for item in positions
                ))
            for warning in paper.get("warnings", [])[:2]:
                lines.append(f"> 模拟盘提示：{warning}")
        hedge = strategy_signal.get("index_hedge") or {}
        if hedge:
            lines.extend([
                "",
                "### CSI300 指数对冲模拟盘",
                f"> **组合净值 {hedge.get('nav', 0):,.2f}**｜股票净值 {hedge.get('stock_nav', 0):,.2f}｜对冲累计损益 {hedge.get('hedge_equity', 0):+,.2f}",
                f"> 当前已生效 {hedge.get('active_hedge_ratio', 0):.0%}｜**下一交易日目标 {hedge.get('target_hedge_ratio', 0):.0%}**｜今日对冲损益 {hedge.get('hedge_pnl_today', 0):+,.2f}",
                f"> CSI300 {hedge.get('index_close', 0):.2f}｜MA{hedge.get('parameters', {}).get('lookback', 120)} {hedge.get('index_ma', 0):.2f}｜趋势比 {hedge.get('index_trend_ratio', 0):.4f}",
                f"> {hedge.get('action_label', '--')}；信号收盘生成、下一交易日开盘执行。",
                "> 仅记录股指期货代理模拟，不连接券商；未计基差、保证金、展期和融资成本。",
            ])
        curve = strategy_signal.get("paper_curve") or {}
        if curve:
            lines.extend([
                "",
                "### 模拟盘收益曲线",
                f"> 前向观察 {curve.get('observations', 0)} 日｜累计收益 {curve.get('composite_return', 0):+.2%}｜最大回撤 {curve.get('max_drawdown', 0):.2%}",
                "> 收益曲线图片随本消息单独发送；仅统计新模拟盘每日快照，不回填历史回测。",
            ])
        decision = strategy_signal.get("decision") or {}
        if decision:
            action = decision.get("action_label", "继续持有")
            execution = decision.get("execution_date", "无交易")
            lines.extend([
                "",
                "### 每日自适应交易决策",
                f"> **今日动作：{action}**｜{execution}",
                f"> 信号日：{decision.get('signal_date', '--')}；周期第 {decision.get('cycle_day', 0)} / {decision.get('rules', {}).get('rebalance_days', 10)} 个交易日；距主调仓 {decision.get('next_scheduled_in', '--')} 个交易日。",
                f"> 原因：{decision.get('reason', '--')}",
            ])
            executed = decision.get("executed_order") or {}
            if executed:
                lines.append(
                    f"> 今日开盘已执行：{executed.get('action_label', '--')}（源自 {executed.get('signal_date', '--')} 收盘信号）"
                )
            if decision.get("trade_required"):
                sells = "、".join(f"{item['name']}({item['code']})" for item in decision.get("sells", [])) or "无"
                buys = "、".join(f"{item['name']}({item['code']})" for item in decision.get("buys", [])) or "无新增成分，仅恢复等权"
                lines.extend([f"卖出：{sells}", f"买入：{buys}"])
            else:
                lines.append("**今日不交易，继续持有。**")
            portfolio = decision.get("target_holdings") or decision.get("current_holdings") or []
            if portfolio:
                lines.append("目标持仓：" + "、".join(
                    f"{item['name']}({item['code']}) {item.get('target_weight', 0):.0%}" for item in portfolio
                ))
        if strategy_picks:
            lines.extend([
                "",
                f"### 今日观察 Top{len(strategy_picks)}（不等于交易指令）",
                f"> 信号日：{strategy_signal.get('signal_date', '--')}；每日排名用于自适应判断，只有上方决策触发时才交易。",
                "",
            ])
            lines.extend(
                f"{item.rank}. **{item.name}（{item.code}）**｜{item.score:.1f}分｜目标 {item.target_weight:.0%}｜参考价 {item.price:.2f}｜{item.reason}"
                for item in strategy_picks
            )
        lines.extend(["", "候选股票："])
        lines.extend(
            f"{item.rank}. {item.name}({item.code}) {item.total_score:.1f}分 | {item.board or item.industry}{'·科技' if item.is_technology else ''} | {item.pct_change:+.2f}%"
            for item in candidates[:12]
        )
        technology = [item for item in candidates if item.is_technology][:5]
        if technology:
            lines.extend(["", "科创/科技观察："])
            lines.extend(
                f"{item.name}({item.code}) {item.total_score:.1f}分 | {item.board or item.industry}"
                for item in technology
            )
        lines.extend(["", "完整研报已存入研究工作台。", "仅供量化研究，不构成投资建议。"])
        return "\n".join(lines)

    @staticmethod
    def _post(channel: str, url: str, payload: Dict) -> Dict:
        try:
            with httpx.Client(timeout=20, trust_env=False) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                try:
                    body = response.json()
                except ValueError:
                    body = None
                if isinstance(body, dict):
                    error_code = body.get("errcode", body.get("code", 0))
                    if error_code not in {0, "0", None}:
                        raise RuntimeError(
                            f"Webhook返回错误 {error_code}: "
                            f"{body.get('errmsg') or body.get('msg') or body}"
                        )
                return {"channel": channel, "ok": True, "detail": "推送成功"}
        except Exception as exc:
            return {"channel": channel, "ok": False, "detail": str(exc)}

    def _email(self, title: str, report: str) -> Dict:
        try:
            message = EmailMessage()
            message["Subject"] = title
            message["From"] = self.settings.email_from or self.settings.smtp_username
            message["To"] = self.settings.email_to
            message.set_content(report)
            with smtplib.SMTP_SSL(self.settings.smtp_host, self.settings.smtp_port, timeout=30) as server:
                if self.settings.smtp_username:
                    server.login(self.settings.smtp_username, self.settings.smtp_password)
                server.send_message(message)
            return {"channel": "邮件", "ok": True, "detail": "推送成功"}
        except Exception as exc:
            return {"channel": "邮件", "ok": False, "detail": str(exc)}
