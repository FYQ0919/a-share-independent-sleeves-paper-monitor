from html.parser import HTMLParser
import sqlite3
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR
from app.local_auth import LocalWebAuth


class CsrfParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.token = ""

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "input" and values.get("name") == "csrf_token":
            self.token = values.get("value", "")


def csrf_token(response):
    parser = CsrfParser()
    parser.feed(response.text)
    assert parser.token
    return parser.token


def local_settings(tmp_path, **overrides):
    values = {
        "web_auth_provider": "local",
        "local_registration_enabled": True,
        "local_registration_invite_code": "invite-only",
        "auth_password_min_length": 10,
        "auth_lockout_attempts": 3,
        "auth_lockout_minutes": 15,
        "auth_database_path": tmp_path / "auth.db",
        "session_secret": "s" * 40,
        "session_max_age_seconds": 43_200,
        "session_cookie_secure": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def build_app(tmp_path, **overrides):
    app = FastAPI()
    auth = LocalWebAuth(
        local_settings(tmp_path, **overrides),
        Jinja2Templates(directory=BASE_DIR / "templates"),
    )
    auth.install(app)

    @app.get("/private")
    async def private_page():
        return {"ok": True}

    @app.get("/api/private")
    async def private_api():
        return {"ok": True}

    @app.post("/api/write")
    async def write_api():
        return {"written": True}

    @app.get("/healthz")
    async def health():
        return {"ok": True}

    return app, auth


def register(client, **overrides):
    page = client.get("/auth/register")
    data = {
        "csrf_token": csrf_token(page),
        "username": "researcher01",
        "display_name": "研究用户",
        "invite_code": "invite-only",
        "password": "unique-passphrase-2026",
        "password_confirm": "unique-passphrase-2026",
    }
    data.update(overrides)
    return client.post("/auth/register", data=data, follow_redirects=False)


def test_registration_creates_hashed_account_and_authenticated_session(tmp_path):
    app, auth = build_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/private", follow_redirects=False).status_code == 303
        assert client.get("/api/private").status_code == 401
        response = register(client)
        assert response.status_code == 303
        assert response.headers["location"] == "/paper-monitor"
        assert client.get("/private").json() == {"ok": True}
        me = client.get("/api/auth/me").json()
        assert me["user"]["username"] == "researcher01"
        assert me["user"]["is_admin"] == 1

    with sqlite3.connect(auth.store.path) as connection:
        row = connection.execute(
            "SELECT password_salt, password_hash FROM users WHERE username = ?",
            ("researcher01",),
        ).fetchone()
    assert row
    assert "unique-passphrase-2026" not in row
    assert row[0] != row[1]


def test_logout_then_password_login_round_trip(tmp_path):
    app, _ = build_app(tmp_path)
    with TestClient(app) as client:
        assert register(client).status_code == 303
        assert client.post("/auth/logout", follow_redirects=False).status_code == 303
        assert client.get("/private", follow_redirects=False).status_code == 303
        login_page = client.get("/auth/login?next=/private")
        response = client.post(
            "/auth/login",
            data={
                "csrf_token": csrf_token(login_page),
                "next": "/private",
                "username": "Researcher01",
                "password": "unique-passphrase-2026",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/private"
        assert client.get("/private").status_code == 200


def test_registration_rejects_invalid_csrf_invite_and_duplicate_username(tmp_path):
    app, _ = build_app(tmp_path)
    with TestClient(app) as client:
        assert client.post(
            "/auth/register",
            data={"csrf_token": "forged"},
            follow_redirects=False,
        ).status_code == 400
        assert register(client, invite_code="wrong").status_code == 403
        assert register(client).status_code == 303
        client.post("/auth/logout", follow_redirects=False)
        assert register(client, username="RESEARCHER01").status_code == 409


def test_repeated_password_failures_lock_account(tmp_path):
    app, _ = build_app(tmp_path)
    with TestClient(app) as client:
        assert register(client).status_code == 303
        client.post("/auth/logout", follow_redirects=False)
        statuses = []
        for _ in range(3):
            page = client.get("/auth/login")
            response = client.post(
                "/auth/login",
                data={
                    "csrf_token": csrf_token(page),
                    "next": "/",
                    "username": "researcher01",
                    "password": "wrong-password",
                },
                follow_redirects=False,
            )
            statuses.append(response.status_code)
        page = client.get("/auth/login")
        locked = client.post(
            "/auth/login",
            data={
                "csrf_token": csrf_token(page),
                "next": "/",
                "username": "researcher01",
                "password": "unique-passphrase-2026",
            },
            follow_redirects=False,
        )
    assert statuses == [401, 401, 401]
    assert locked.status_code == 401
    assert "暂时锁定" in locked.text


def test_registration_can_be_disabled_and_configuration_fails_closed(tmp_path):
    app, auth = build_app(
        tmp_path,
        local_registration_enabled=False,
        session_secret="short",
    )
    with TestClient(app) as client:
        assert client.get("/auth/login").status_code == 503
        assert client.get("/auth/register").status_code == 503
        assert client.post("/auth/register", data={}).status_code == 503
        assert client.get("/private").status_code == 503
    assert auth.status.ready is False
    assert "32" in auth.status.errors[0]


def test_registration_disabled_returns_404_when_auth_is_ready(tmp_path):
    app, _ = build_app(tmp_path, local_registration_enabled=False)
    with TestClient(app) as client:
        assert client.get("/auth/register").status_code == 404
        assert client.post("/auth/register", data={}).status_code == 404


def test_first_user_is_admin_and_later_users_are_read_only(tmp_path):
    app, _ = build_app(tmp_path)
    with TestClient(app) as first:
        assert register(first, username="first-admin").status_code == 303
        assert first.post("/api/write").status_code == 200
    with TestClient(app) as second:
        assert register(
            second,
            username="viewer-user",
            display_name="只读用户",
        ).status_code == 303
        assert second.get("/api/private").status_code == 200
        denied = second.post("/api/write")
        assert denied.status_code == 403
        assert "只读" in denied.json()["detail"]


def test_open_registration_never_grants_admin_implicitly(tmp_path):
    app, _ = build_app(tmp_path, local_registration_invite_code="")
    with TestClient(app) as client:
        assert register(
            client,
            invite_code="",
            username="open-user",
        ).status_code == 303
        assert client.get("/api/auth/me").json()["user"]["is_admin"] == 0
        assert client.post("/api/write").status_code == 403


def test_password_reset_invalidates_existing_sessions(tmp_path):
    app, auth = build_app(tmp_path)
    with TestClient(app) as client:
        assert register(client).status_code == 303
        assert client.get("/private").status_code == 200
        assert auth.store.reset_password("researcher01", "another-unique-pass-2026")
        assert client.get("/private", follow_redirects=False).status_code == 303
