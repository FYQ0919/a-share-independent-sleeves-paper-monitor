from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import hmac
import re
import secrets
import sqlite3
from pathlib import Path
from threading import Lock
import time
from typing import Any, Mapping
from urllib.parse import parse_qs, urlencode
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.templating import Jinja2Templates

from app.web_auth import AuthStatus, SignedCookie, _safe_next


SESSION_COOKIE = "quant_local_session"
CSRF_COOKIE = "quant_local_csrf"
USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
COMMON_PASSWORDS = {
    "1234567890",
    "password123",
    "qwerty12345",
    "1111111111",
    "admin123456",
}


def _utc_now() -> int:
    return int(time.time())


def _normalize_username(value: str) -> str:
    return value.strip().lower()


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    password_salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=password_salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )
    return (
        base64.b64encode(password_salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def _verify_password(password: str, salt_text: str, expected_text: str) -> bool:
    try:
        salt = base64.b64decode(salt_text, validate=True)
        _, actual = _hash_password(password, salt)
        return hmac.compare_digest(actual, expected_text)
    except (ValueError, TypeError):
        return False


class AccountStore:
    def __init__(self, path: Path, lockout_attempts: int = 5, lockout_minutes: int = 15):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lockout_attempts = max(2, int(lockout_attempts))
        self.lockout_seconds = max(1, int(lockout_minutes)) * 60
        self._lock = Lock()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    session_version INTEGER NOT NULL DEFAULT 1,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_login_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_hash TEXT NOT NULL,
                    action TEXT NOT NULL,
                    attempted_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_auth_events_lookup "
                "ON auth_events(source_hash, action, attempted_at)"
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(users)").fetchall()
            }
            if "session_version" not in columns:
                connection.execute(
                    "ALTER TABLE users ADD COLUMN session_version INTEGER NOT NULL DEFAULT 1"
                )

    @staticmethod
    def validate_registration(
        username: str, display_name: str, password: str, password_min_length: int
    ) -> list[str]:
        errors = []
        if not USERNAME_PATTERN.fullmatch(username):
            errors.append("用户名需为3-32位小写字母、数字、点、下划线或连字符")
        if not 1 <= len(display_name.strip()) <= 40:
            errors.append("显示名称需为1-40个字符")
        if len(password) < password_min_length or len(password) > 128:
            errors.append(f"密码需为{password_min_length}-128个字符")
        if password.lower() in COMMON_PASSWORDS or password.lower() == username:
            errors.append("密码过于常见，请使用更长且唯一的密码")
        return errors

    def create_user(
        self,
        username: str,
        display_name: str,
        password: str,
        admin_if_first: bool = False,
    ) -> dict[str, Any] | None:
        normalized = _normalize_username(username)
        salt, password_hash = _hash_password(password)
        user_id = uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                is_first = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
                connection.execute(
                    """
                    INSERT INTO users(
                        user_id, username, display_name, password_salt, password_hash,
                        is_active, is_admin, created_at
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        user_id,
                        normalized,
                        display_name.strip(),
                        salt,
                        password_hash,
                        int(is_first and admin_if_first),
                        created_at,
                    ),
                )
        except sqlite3.IntegrityError:
            return None
        return self.get_user(user_id)

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT user_id, username, display_name, is_active, is_admin, session_version, "
                "created_at, last_login_at FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row and row["is_active"] else None

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT user_id, username, display_name, is_active, is_admin, session_version, "
                "created_at, last_login_at FROM users ORDER BY created_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def set_active(self, username: str, active: bool) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE users SET is_active = ? WHERE username = ?",
                (int(active), _normalize_username(username)),
            )
        return cursor.rowcount == 1

    def set_admin(self, username: str, admin: bool) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE users SET is_admin = ? WHERE username = ?",
                (int(admin), _normalize_username(username)),
            )
        return cursor.rowcount == 1

    def reset_password(self, username: str, password: str) -> bool:
        salt, password_hash = _hash_password(password)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE users SET password_salt = ?, password_hash = ?, "
                "failed_attempts = 0, locked_until = 0, session_version = session_version + 1 "
                "WHERE username = ?",
                (salt, password_hash, _normalize_username(username)),
            )
        return cursor.rowcount == 1

    def authenticate(self, username: str, password: str) -> tuple[dict[str, Any] | None, str]:
        normalized = _normalize_username(username)
        now = _utc_now()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ?", (normalized,)
            ).fetchone()
            if not row:
                _hash_password(password, b"quant-dummy-salt")
                return None, "invalid"
            if not row["is_active"]:
                return None, "invalid"
            if int(row["locked_until"] or 0) > now:
                return None, "locked"
            if not _verify_password(password, row["password_salt"], row["password_hash"]):
                attempts = int(row["failed_attempts"] or 0) + 1
                locked_until = now + self.lockout_seconds if attempts >= self.lockout_attempts else 0
                connection.execute(
                    "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE user_id = ?",
                    (attempts, locked_until, row["user_id"]),
                )
                return None, "locked" if locked_until else "invalid"
            last_login = datetime.now(timezone.utc).isoformat()
            connection.execute(
                "UPDATE users SET failed_attempts = 0, locked_until = 0, last_login_at = ? "
                "WHERE user_id = ?",
                (last_login, row["user_id"]),
            )
        return self.get_user(row["user_id"]), "ok"

    def rate_limited(
        self, source: str, action: str, limit: int, window_seconds: int
    ) -> bool:
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        now = _utc_now()
        cutoff = now - window_seconds
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM auth_events WHERE attempted_at < ?", (cutoff,))
            count = connection.execute(
                "SELECT COUNT(*) FROM auth_events WHERE source_hash = ? AND action = ? "
                "AND attempted_at >= ?",
                (source_hash, action, cutoff),
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO auth_events(source_hash, action, attempted_at) VALUES (?, ?, ?)",
                (source_hash, action, now),
            )
        return count >= limit


class LocalAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, auth: "LocalWebAuth"):
        super().__init__(app)
        self.auth = auth

    async def dispatch(self, request: Request, call_next):
        if self.auth.is_public_path(request.url.path):
            return await call_next(request)
        if not self.auth.status.ready:
            payload = {"detail": "账号登录尚未安全配置完成"}
            if request.url.path.startswith("/api/"):
                return JSONResponse(payload, status_code=503)
            return HTMLResponse(payload["detail"], status_code=503)
        user = self.auth.read_user(request)
        if user:
            request.state.auth_user = user
            is_write = request.method.upper() not in {"GET", "HEAD", "OPTIONS"}
            if is_write and request.url.path != "/auth/logout" and not user["is_admin"]:
                if request.url.path.startswith("/api/"):
                    return JSONResponse(
                        {"detail": "只读账号不能执行此操作"}, status_code=403
                    )
                return HTMLResponse("只读账号不能执行此操作", status_code=403)
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                {
                    "detail": "请先登录",
                    "login_url": f"/auth/login?{urlencode({'next': request.url.path})}",
                },
                status_code=401,
            )
        destination = request.url.path
        if request.url.query:
            destination = f"{destination}?{request.url.query}"
        return RedirectResponse(
            f"/auth/login?{urlencode({'next': destination})}", status_code=303
        )


class LocalWebAuth:
    def __init__(self, settings, templates: Jinja2Templates):
        self.settings = settings
        self.templates = templates
        self.enabled = settings.web_auth_provider == "local"
        self.logout_path = "/auth/logout"
        self.signer = SignedCookie(settings.session_secret or secrets.token_urlsafe(32))
        self.store = AccountStore(
            settings.auth_database_path,
            settings.auth_lockout_attempts,
            settings.auth_lockout_minutes,
        )

    @property
    def status(self) -> AuthStatus:
        errors = []
        if not self.settings.session_secret:
            errors.append("缺少 SESSION_SECRET")
        elif len(self.settings.session_secret) < 32:
            errors.append("SESSION_SECRET 至少需要 32 个字符")
        return AuthStatus(True, not errors, errors)

    @staticmethod
    def is_public_path(path: str) -> bool:
        return path == "/healthz" or path.startswith("/static/") or path in {
            "/auth/login",
            "/auth/register",
        }

    def _set_cookie(self, response, name: str, value: str, max_age: int) -> None:
        response.set_cookie(
            name,
            value,
            max_age=max_age,
            httponly=True,
            secure=self.settings.session_cookie_secure,
            samesite="lax",
            path="/",
        )

    def _form_response(
        self,
        request: Request,
        template: str,
        context: Mapping[str, Any],
        status_code: int = 200,
    ):
        csrf = self.signer.dumps({"purpose": "account_form"}, max_age=3600)
        response = self.templates.TemplateResponse(
            request=request,
            name=template,
            context={"csrf_token": csrf, **context},
            status_code=status_code,
            headers={"Cache-Control": "no-store"},
        )
        self._set_cookie(response, CSRF_COOKIE, csrf, max_age=3600)
        return response

    def _valid_csrf(self, request: Request, token: str) -> bool:
        cookie = request.cookies.get(CSRF_COOKIE, "")
        if not cookie or not token or not hmac.compare_digest(cookie, token):
            return False
        payload = self.signer.loads(token)
        return bool(payload and payload.get("purpose") == "account_form")

    def read_user(self, request: Request) -> dict[str, Any] | None:
        payload = self.signer.loads(request.cookies.get(SESSION_COOKIE))
        if not payload or not payload.get("user_id"):
            return None
        user = self.store.get_user(str(payload["user_id"]))
        if not user or int(payload.get("session_version", 0)) != int(
            user["session_version"]
        ):
            return None
        return user

    @staticmethod
    async def _form_data(request: Request) -> dict[str, str]:
        body = await request.body()
        if len(body) > 16_384:
            return {}
        try:
            text = body.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return {}
        parsed = parse_qs(text, keep_blank_values=True)
        return {key: values[-1] for key, values in parsed.items() if values}

    @staticmethod
    def _source(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def install(self, app: FastAPI) -> None:
        auth = self

        @app.get("/auth/login", response_class=HTMLResponse)
        async def login_page(request: Request, next: str = "/"):
            if not auth.status.ready:
                return HTMLResponse(
                    "账号登录尚未配置完成：" + "；".join(auth.status.errors),
                    status_code=503,
                )
            if auth.read_user(request):
                return RedirectResponse(_safe_next(next), status_code=303)
            return auth._form_response(
                request,
                "login.html",
                {
                    "next_path": _safe_next(next),
                    "error": "",
                    "registration_enabled": auth.settings.local_registration_enabled,
                },
            )

        @app.post("/auth/login", response_class=HTMLResponse)
        async def login_submit(request: Request):
            if not auth.status.ready:
                return HTMLResponse("账号登录尚未安全配置完成", status_code=503)
            form = await auth._form_data(request)
            next_path = _safe_next(form.get("next"))
            if not auth._valid_csrf(request, form.get("csrf_token", "")):
                return HTMLResponse("登录表单已过期，请刷新后重试", status_code=400)
            if auth.store.rate_limited(auth._source(request), "login", 20, 600):
                return auth._form_response(
                    request,
                    "login.html",
                    {
                        "next_path": next_path,
                        "error": "尝试次数过多，请稍后再试",
                        "registration_enabled": auth.settings.local_registration_enabled,
                    },
                    status_code=429,
                )
            user, result = auth.store.authenticate(
                form.get("username", ""), form.get("password", "")
            )
            if not user:
                message = "账号暂时锁定，请稍后再试" if result == "locked" else "用户名或密码错误"
                return auth._form_response(
                    request,
                    "login.html",
                    {
                        "next_path": next_path,
                        "error": message,
                        "registration_enabled": auth.settings.local_registration_enabled,
                    },
                    status_code=401,
                )
            session = auth.signer.dumps(
                {
                    "user_id": user["user_id"],
                    "session_version": user["session_version"],
                },
                max_age=auth.settings.session_max_age_seconds,
            )
            response = RedirectResponse(next_path, status_code=303)
            auth._set_cookie(
                response,
                SESSION_COOKIE,
                session,
                max_age=auth.settings.session_max_age_seconds,
            )
            response.delete_cookie(CSRF_COOKIE, path="/")
            return response

        @app.get("/auth/register", response_class=HTMLResponse)
        async def register_page(request: Request):
            if not auth.status.ready:
                return HTMLResponse("账号登录尚未安全配置完成", status_code=503)
            if not auth.settings.local_registration_enabled:
                return HTMLResponse("网站暂未开放注册", status_code=404)
            if auth.read_user(request):
                return RedirectResponse("/", status_code=303)
            return auth._form_response(
                request,
                "register.html",
                {
                    "error": "",
                    "invite_required": bool(auth.settings.local_registration_invite_code),
                    "password_min_length": auth.settings.auth_password_min_length,
                },
            )

        @app.post("/auth/register", response_class=HTMLResponse)
        async def register_submit(request: Request):
            if not auth.status.ready:
                return HTMLResponse("账号登录尚未安全配置完成", status_code=503)
            if not auth.settings.local_registration_enabled:
                return HTMLResponse("网站暂未开放注册", status_code=404)
            form = await auth._form_data(request)
            context = {
                "error": "",
                "invite_required": bool(auth.settings.local_registration_invite_code),
                "password_min_length": auth.settings.auth_password_min_length,
            }
            if not auth._valid_csrf(request, form.get("csrf_token", "")):
                return HTMLResponse("注册表单已过期，请刷新后重试", status_code=400)
            if auth.store.rate_limited(auth._source(request), "register", 5, 600):
                return auth._form_response(
                    request,
                    "register.html",
                    {**context, "error": "注册次数过多，请稍后再试"},
                    status_code=429,
                )
            required_invite = auth.settings.local_registration_invite_code
            if required_invite and not hmac.compare_digest(
                form.get("invite_code", ""), required_invite
            ):
                return auth._form_response(
                    request,
                    "register.html",
                    {**context, "error": "邀请码无效"},
                    status_code=403,
                )
            username = _normalize_username(form.get("username", ""))
            display_name = form.get("display_name", "").strip()
            password = form.get("password", "")
            if password != form.get("password_confirm", ""):
                return auth._form_response(
                    request,
                    "register.html",
                    {**context, "error": "两次输入的密码不一致"},
                    status_code=400,
                )
            errors = auth.store.validate_registration(
                username,
                display_name,
                password,
                auth.settings.auth_password_min_length,
            )
            if errors:
                return auth._form_response(
                    request,
                    "register.html",
                    {**context, "error": "；".join(errors)},
                    status_code=400,
                )
            user = auth.store.create_user(
                username,
                display_name,
                password,
                admin_if_first=bool(required_invite),
            )
            if not user:
                return auth._form_response(
                    request,
                    "register.html",
                    {**context, "error": "该用户名已经存在"},
                    status_code=409,
                )
            session = auth.signer.dumps(
                {
                    "user_id": user["user_id"],
                    "session_version": user["session_version"],
                },
                max_age=auth.settings.session_max_age_seconds,
            )
            response = RedirectResponse("/paper-monitor", status_code=303)
            auth._set_cookie(
                response,
                SESSION_COOKIE,
                session,
                max_age=auth.settings.session_max_age_seconds,
            )
            response.delete_cookie(CSRF_COOKIE, path="/")
            return response

        @app.get("/api/auth/me")
        async def auth_me(request: Request):
            user = auth.read_user(request)
            return {"enabled": True, "provider": "local", "authenticated": bool(user), "user": user}

        @app.post("/auth/logout")
        async def logout():
            response = RedirectResponse("/auth/login", status_code=303)
            response.delete_cookie(SESSION_COOKIE, path="/")
            response.delete_cookie(CSRF_COOKIE, path="/")
            return response

        app.add_middleware(LocalAuthMiddleware, auth=self)
