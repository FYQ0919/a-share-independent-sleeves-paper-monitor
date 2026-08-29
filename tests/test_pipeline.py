from dataclasses import replace

from app.config import settings
from app.pipeline import ResearchPipeline
from app.storage import Storage


def test_demo_pipeline_generates_and_persists_report(tmp_path):
    test_settings = replace(
        settings,
        database_path=tmp_path / "research.db",
        report_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
        llm_api_key="",
        feishu_webhook_url="",
        wecom_webhook_url="",
        generic_webhook_url="",
        smtp_host="",
        email_to="",
    )
    storage = Storage(test_settings.database_path)
    result = ResearchPipeline(test_settings, storage).run("demo", notify=False)

    assert result.mode == "demo"
    assert len(result.candidates) == test_settings.top_n
    assert "市场概览" in result.report_markdown
    assert "宏观环境评测" in result.report_markdown
    assert "基础策略保护：优先执行基础策略" in result.report_markdown
    assert result.macro_environment["automatic_execution"] is False
    assert result.macro_environment["regime"] == "中性"
    assert "板块强弱" in result.report_markdown
    assert "Top 候选" in result.report_markdown
    assert storage.latest()["run_id"] == result.run_id
    assert storage.latest()["macro_environment"]["version"] == "macro-overlay-v2-base-protected"
    assert list(test_settings.report_dir.glob("*_research.md"))


def test_unconfigured_notification_is_explicit(tmp_path):
    test_settings = replace(
        settings,
        database_path=tmp_path / "research.db",
        report_dir=tmp_path / "reports",
        cache_dir=tmp_path / "cache",
        llm_api_key="",
        feishu_webhook_url="",
        wecom_webhook_url="",
        generic_webhook_url="",
        smtp_host="",
        email_to="",
    )
    result = ResearchPipeline(test_settings, Storage(test_settings.database_path)).run("demo", notify=True)
    assert result.notifications[0]["channel"] == "未配置"
    assert result.notifications[0]["ok"] is False
