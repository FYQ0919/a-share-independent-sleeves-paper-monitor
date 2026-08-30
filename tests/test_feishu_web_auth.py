from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR
from app.web_auth import FeishuWebAuth, _safe_next


def auth_settings(**overrides):
    values = {
        "feishu_web_login_enabled": True,
        "web_auth_provider": "feishu",
        "feishu_app_id": "cli_test",
        "feishu_app_secret": "test-secret",
        "feishu_redirect_uri": "https://quant.example.com/auth/feishu/callback",
        "feishu_allowed_open_ids": ("ou_allowed",),
        "feishu_allowed_tenant_keys": (),
        "feishu_allow_any_authenticated": False,
        "session_secret": "s" * 40,
        "session_max_age_seconds": 43_200,
        "session_cookie_secure": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeOAuthClient:
    def __init__(self, user=None):
        self.user = user or {
            "open_id": "ou_allowed",
            "tenant_key": "tenant_a",
            "name": "测试用户",
        }
        self.codes = []

    async def authenticate(self, code):
        self.codes.append(code)
        return self.user


def build_app(settings=None, oauth_client=None):
    app = FastAPI()
    templates = Jinja2Templates(directory=BASE_DIR / "templates")
    auth = FeishuWebAuth(settings or auth_settings(), templates)
    auth.oauth_client = oauth_client or FakeOAuthClient()
    auth.install(app)

    @app.get("/private")
    async def private_page():
        return {"ok": True}

    @app.get("/api/private")
    async def private_api():
        return {"ok": True}

    @app.get("/healthz")
    async def health():
        return {"ok": True}

    return app, auth


def test_protected_pages_redirect_and_apis_return_401():
    app, _ = build_app()
    with TestClient(app) as client:
        page = client.get("/private?view=full", follow_redirects=False)
        api = client.get("/api/private", follow_redirects=False)
        health = client.get("/healthz")

    assert page.status_code == 303
    assert page.headers["location"].startswith("/auth/feishu/login?")
    assert api.status_code == 401
    assert api.json()["login_url"].endswith("next=%2Fapi%2Fprivate")
    assert health.status_code == 200


def test_full_login_callback_sets_session_and_logout_clears_it():
    oauth = FakeOAuthClient()
    app, _ = build_app(oauth_client=oauth)
    with TestClient(app) as client:
        start = client.get(
            "/auth/feishu/start?next=/private", follow_redirects=False
        )
        assert start.status_code == 303
        parsed = urlparse(start.headers["location"])
        state = parse_qs(parsed.query)["state"][0]

        callback = client.get(
            f"/auth/feishu/callback?code=code-1&state={state}",
            follow_redirects=False,
        )
        assert callback.status_code == 303
        assert callback.headers["location"] == "/private"
        assert client.get("/private").json() == {"ok": True}
        me = client.get("/api/auth/me").json()
        assert me["authenticated"] is True
        assert me["user"]["name"] == "测试用户"

        logout = client.post("/auth/feishu/logout", follow_redirects=False)
        assert logout.status_code == 303
        assert client.get("/private", follow_redirects=False).status_code == 303
    assert oauth.codes == ["code-1"]


def test_callback_rejects_wrong_state_and_user_outside_allowlist():
    denied_oauth = FakeOAuthClient(
        {"open_id": "ou_denied", "tenant_key": "tenant_b", "name": "无权限用户"}
    )
    app, _ = build_app(oauth_client=denied_oauth)
    with TestClient(app) as client:
        assert client.get(
            "/auth/feishu/callback?code=x&state=wrong",
            follow_redirects=False,
        ).status_code == 400
        start = client.get("/auth/feishu/start", follow_redirects=False)
        state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
        denied = client.get(
            f"/auth/feishu/callback?code=x&state={state}", follow_redirects=False
        )
        assert denied.status_code == 403


def test_incomplete_configuration_fails_closed_and_unsafe_next_is_removed():
    settings = auth_settings(session_secret="short", feishu_allowed_open_ids=())
    app, auth = build_app(settings=settings)
    with TestClient(app) as client:
        login = client.get(
            "/auth/feishu/login?next=//outside.example", follow_redirects=False
        )
        start = client.get("/auth/feishu/start", follow_redirects=False)

    assert auth.status.ready is False
    assert login.status_code == 503
    assert "outside.example" not in login.text
    assert "/auth/feishu/start?" not in login.text
    assert start.status_code == 503


def test_next_path_rejects_browser_normalization_edge_cases():
    assert _safe_next("/paper-monitor?view=full") == "/paper-monitor?view=full"
    assert _safe_next("//outside.example") == "/"
    assert _safe_next("/\\outside.example") == "/"
    assert _safe_next("/paper-monitor\nLocation: https://outside.example") == "/"
