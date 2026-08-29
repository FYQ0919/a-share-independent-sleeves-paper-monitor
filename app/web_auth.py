from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlencode, urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.templating import Jinja2Templates


FEISHU_AUTHORIZE_URL = "https://accounts.feishu.cn/open-apis/authen/v1/authorize"
FEISHU_TOKEN_URL = "https://open.feishu.cn/open-apis/authen/v2/oauth/token"
FEISHU_USER_INFO_URL = "https://open.feishu.cn/open-apis/authen/v1/user_info"
SESSION_COOKIE = "quant_feishu_session"
OAUTH_COOKIE = "quant_feishu_oauth"


def _safe_next(value: str | None) -> str:
    if (
        not value
        or not value.startswith("/")
        or value.startswith("//")
        or "\\" in value
        or any(ord(char) < 32 for char in value)
    ):
        return "/"
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return "/"
    return value


class SignedCookie:
    def __init__(self, secret: str):
        self.secret = secret.encode("utf-8")

    @staticmethod
    def _encode(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def _decode(value: str) -> bytes:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

    def dumps(self, payload: Mapping[str, Any], max_age: int) -> str:
        body = dict(payload)
        body["exp"] = int(time.time()) + max_age
        encoded = self._encode(
            json.dumps(body, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        signature = self._encode(
            hmac.new(self.secret, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        return f"{encoded}.{signature}"

    def loads(self, value: str | None) -> dict[str, Any] | None:
        if not value:
            return None
        try:
            encoded, signature = value.split(".", 1)
            expected = self._encode(
                hmac.new(self.secret, encoded.encode("ascii"), hashlib.sha256).digest()
            )
            if not hmac.compare_digest(signature, expected):
                return None
            payload = json.loads(self._decode(encoded).decode("utf-8"))
            if int(payload.get("exp", 0)) < int(time.time()):
                return None
            return payload
        except (ValueError, TypeError, json.JSONDecodeError):
            return None


class FeishuOAuthClient:
    def __init__(self, app_id: str, app_secret: str, redirect_uri: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.redirect_uri = redirect_uri

    async def authenticate(self, code: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
            token_response = await client.post(
                FEISHU_TOKEN_URL,
                json={
                    "grant_type": "authorization_code",
                    "client_id": self.app_id,
                    "client_secret": self.app_secret,
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                },
            )
            token_response.raise_for_status()
            token_payload = token_response.json()
            access_token = token_payload.get("access_token")
            if not access_token:
                detail = token_payload.get("error_description") or token_payload.get("msg")
                raise RuntimeError(detail or "飞书没有返回 user_access_token")
            user_response = await client.get(
                FEISHU_USER_INFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            user_response.raise_for_status()
            user_payload = user_response.json()
        if user_payload.get("code") not in (None, 0):
            raise RuntimeError(user_payload.get("msg") or "飞书用户信息读取失败")
        return user_payload.get("data") or user_payload


@dataclass
class AuthStatus:
    enabled: bool
    ready: bool
    errors: list[str]


class FeishuAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, auth: "FeishuWebAuth"):
        super().__init__(app)
        self.auth = auth

    async def dispatch(self, request: Request, call_next):
        if not self.auth.enabled or self.auth.is_public_path(request.url.path):
            return await call_next(request)
        user = self.auth.read_user(request)
        if user:
            request.state.feishu_user = user
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                {
                    "detail": "请先使用飞书登录",
                    "login_url": f"/auth/feishu/login?{urlencode({'next': request.url.path})}",
                },
                status_code=401,
            )
        destination = request.url.path
        if request.url.query:
            destination = f"{destination}?{request.url.query}"
        return RedirectResponse(
            f"/auth/feishu/login?{urlencode({'next': destination})}",
            status_code=303,
        )


class FeishuWebAuth:
    def __init__(self, settings, templates: Jinja2Templates):
        self.settings = settings
        self.templates = templates
        self.enabled = bool(settings.feishu_web_login_enabled)
        self.signer = SignedCookie(settings.session_secret or secrets.token_urlsafe(32))
        self.oauth_client = FeishuOAuthClient(
            settings.feishu_app_id,
            settings.feishu_app_secret,
            settings.feishu_redirect_uri,
        )

    @property
    def status(self) -> AuthStatus:
        errors = []
        if self.enabled:
            for variable, value in (
                ("FEISHU_APP_ID", self.settings.feishu_app_id),
                ("FEISHU_APP_SECRET", self.settings.feishu_app_secret),
                ("FEISHU_REDIRECT_URI", self.settings.feishu_redirect_uri),
                ("SESSION_SECRET", self.settings.session_secret),
            ):
                if not value:
                    errors.append(f"缺少 {variable}")
            if self.settings.session_secret and len(self.settings.session_secret) < 32:
                errors.append("SESSION_SECRET 至少需要 32 个字符")
            has_policy = bool(
                self.settings.feishu_allow_any_authenticated
                or self.settings.feishu_allowed_open_ids
                or self.settings.feishu_allowed_tenant_keys
            )
            if not has_policy:
                errors.append("尚未配置飞书成员、租户白名单或显式允许任意已登录用户")
        return AuthStatus(self.enabled, self.enabled and not errors, errors)

    @staticmethod
    def is_public_path(path: str) -> bool:
        return path == "/healthz" or path.startswith("/static/") or path.startswith(
            "/auth/feishu/"
        )

    def read_user(self, request: Request) -> dict[str, Any] | None:
        payload = self.signer.loads(request.cookies.get(SESSION_COOKIE))
        if not payload or not payload.get("open_id"):
            return None
        return {
            "open_id": payload["open_id"],
            "tenant_key": payload.get("tenant_key", ""),
            "name": payload.get("name") or "飞书用户",
        }

    def is_allowed(self, user: Mapping[str, Any]) -> bool:
        if self.settings.feishu_allow_any_authenticated:
            return True
        return bool(
            user.get("open_id") in self.settings.feishu_allowed_open_ids
            or user.get("tenant_key") in self.settings.feishu_allowed_tenant_keys
        )

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

    def install(self, app: FastAPI) -> None:
        auth = self

        @app.get("/auth/feishu/login", response_class=HTMLResponse)
        async def feishu_login_page(request: Request, next: str = "/"):
            user = auth.read_user(request)
            if user:
                return RedirectResponse(_safe_next(next), status_code=303)
            status = auth.status
            return auth.templates.TemplateResponse(
                request=request,
                name="login.html",
                context={
                    "next_path": _safe_next(next),
                    "auth_ready": status.ready,
                    "auth_errors": status.errors,
                },
                status_code=200 if status.ready else 503,
                headers={"Cache-Control": "no-store"},
            )

        @app.get("/auth/feishu/start")
        async def feishu_login_start(next: str = "/"):
            status = auth.status
            if not status.ready:
                return HTMLResponse(
                    "飞书登录尚未配置完成：" + "；".join(status.errors), status_code=503
                )
            state = secrets.token_urlsafe(32)
            oauth_cookie = auth.signer.dumps(
                {"state": state, "next": _safe_next(next)}, max_age=600
            )
            query = urlencode(
                {
                    "app_id": auth.settings.feishu_app_id,
                    "redirect_uri": auth.settings.feishu_redirect_uri,
                    "state": state,
                }
            )
            response = RedirectResponse(f"{FEISHU_AUTHORIZE_URL}?{query}", status_code=303)
            auth._set_cookie(response, OAUTH_COOKIE, oauth_cookie, max_age=600)
            response.headers["Cache-Control"] = "no-store"
            return response

        @app.get("/auth/feishu/callback")
        async def feishu_login_callback(
            request: Request,
            code: str = "",
            state: str = "",
            error: str = "",
        ):
            oauth_state = auth.signer.loads(request.cookies.get(OAUTH_COOKIE))
            valid_state = bool(
                oauth_state
                and state
                and hmac.compare_digest(str(oauth_state.get("state", "")), state)
            )
            if error or not code or not valid_state:
                response = HTMLResponse("飞书登录已取消或安全校验失败", status_code=400)
                response.delete_cookie(OAUTH_COOKIE, path="/")
                return response
            try:
                user = await auth.oauth_client.authenticate(code)
            except (httpx.HTTPError, RuntimeError, ValueError):
                response = HTMLResponse("飞书身份验证失败，请重新登录", status_code=502)
                response.delete_cookie(OAUTH_COOKIE, path="/")
                return response
            if not user.get("open_id") or not auth.is_allowed(user):
                response = HTMLResponse("该飞书账号没有查看此网站的权限", status_code=403)
                response.delete_cookie(OAUTH_COOKIE, path="/")
                return response
            session = auth.signer.dumps(
                {
                    "open_id": user["open_id"],
                    "tenant_key": user.get("tenant_key", ""),
                    "name": user.get("name") or "飞书用户",
                },
                max_age=auth.settings.session_max_age_seconds,
            )
            response = RedirectResponse(
                _safe_next(oauth_state.get("next")), status_code=303
            )
            auth._set_cookie(
                response,
                SESSION_COOKIE,
                session,
                max_age=auth.settings.session_max_age_seconds,
            )
            response.delete_cookie(OAUTH_COOKIE, path="/")
            response.headers["Cache-Control"] = "no-store"
            return response

        @app.get("/api/auth/me")
        async def feishu_auth_me(request: Request):
            user = auth.read_user(request)
            return {"enabled": auth.enabled, "authenticated": bool(user), "user": user}

        @app.post("/auth/feishu/logout")
        async def feishu_logout():
            response = RedirectResponse("/auth/feishu/login", status_code=303)
            response.delete_cookie(SESSION_COOKIE, path="/")
            response.delete_cookie(OAUTH_COOKIE, path="/")
            return response

        app.add_middleware(FeishuAuthMiddleware, auth=self)
