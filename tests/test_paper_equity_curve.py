import base64
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from app.config import settings
from app.notifications import NotificationService
from app.paper_curve import PaperEquityCurveService
from app.storage import Storage


def save_snapshot(storage, account_id, signal_date, snapshot):
    state = {
        "account_id": account_id,
        "last_signal_date": signal_date,
        "last_snapshot": snapshot,
    }
    storage.save_strategy_snapshot(account_id, signal_date, state, snapshot)


def test_forward_paper_curve_exports_csv_json_and_png(tmp_path):
    storage = Storage(tmp_path / "paper.db")
    stock_id = "stock_paper"
    hedge_id = "hedge_paper"
    rows = [
        ("2026-08-28", 1_000_000, 1_000_000, 0.0, 0.5),
        ("2026-08-31", 1_020_000, 1_015_000, 0.5, 0.5),
        ("2026-09-01", 1_010_000, 1_012_000, 0.5, 0.0),
    ]
    for signal_date, stock_nav, composite_nav, active, target in rows:
        save_snapshot(
            storage,
            stock_id,
            signal_date,
            {"signal_date": signal_date, "nav": stock_nav},
        )
        save_snapshot(
            storage,
            hedge_id,
            signal_date,
            {
                "signal_date": signal_date,
                "nav": composite_nav,
                "active_hedge_ratio": active,
                "target_hedge_ratio": target,
            },
        )

    result = PaperEquityCurveService(storage, tmp_path / "reports").generate(
        stock_id, hedge_id
    )

    assert result["status"] == "forward_paper_only"
    assert result["observations"] == 3
    assert result["backfilled"] is False
    assert result["composite_return"] == pytest.approx(0.012)
    assert result["max_drawdown"] < 0
    assert result["active_hedge_ratio"] == 0.5
    assert result["target_hedge_ratio"] == 0.0
    for key in ("image_path", "latest_image_path", "json_path", "csv_path"):
        path = Path(result[key])
        assert path.is_file()
    image = Path(result["image_path"]).read_bytes()
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(image) < 2 * 1024 * 1024


def test_wecom_send_posts_markdown_then_native_image(tmp_path, monkeypatch):
    image_path = tmp_path / "curve.png"
    content = b"\x89PNG\r\n\x1a\n" + b"paper-curve"
    image_path.write_bytes(content)
    test_settings = replace(
        settings,
        wecom_webhook_url="https://example.invalid/wecom",
        feishu_webhook_url="",
        generic_webhook_url="",
        smtp_host="",
        email_to="",
    )
    service = NotificationService(test_settings)
    calls = []

    def fake_post(channel, url, payload):
        calls.append((channel, url, payload))
        return {"channel": channel, "ok": True, "detail": "test"}

    monkeypatch.setattr(service, "_post", fake_post)
    results = service.send(
        "paper",
        "summary",
        "report",
        [],
        [],
        strategy_signal={"paper_curve": {"image_path": str(image_path)}},
    )

    assert [item[0] for item in calls] == ["企业微信", "企业微信图片"]
    image_payload = calls[1][2]
    assert image_payload["msgtype"] == "image"
    assert base64.b64decode(image_payload["image"]["base64"]) == content
    assert image_payload["image"]["md5"] == hashlib.md5(content).hexdigest()
    assert all(item["ok"] for item in results)


def test_wecom_image_payload_rejects_non_png(tmp_path):
    path = tmp_path / "not-image.bin"
    path.write_bytes(b"not a png")

    try:
        NotificationService._wecom_image_payload(path)
    except ValueError as exc:
        assert "PNG" in str(exc)
    else:
        raise AssertionError("non-PNG payload should be rejected")
