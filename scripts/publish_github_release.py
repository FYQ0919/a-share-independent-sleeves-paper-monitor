from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


CLIENT_ID = "Iv1.b507a08c87ecfe98"


class Response:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self.body = body

    def json(self) -> dict:
        return json.loads(self.body.decode("utf-8"))

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(
                f"GitHub returned HTTP {self.status_code}: "
                f"{self.body.decode('utf-8', errors='replace')[:500]}"
            )


def request_with_retry(method: str, url: str, **kwargs) -> Response:
    last_error = None
    opener = urllib.request.build_opener()
    for attempt in range(3):
        try:
            headers = dict(kwargs.get("headers", {}))
            body = None
            if kwargs.get("json") is not None:
                body = json.dumps(kwargs["json"]).encode("utf-8")
                headers["Content-Type"] = "application/json"
            elif kwargs.get("data") is not None:
                body = urllib.parse.urlencode(kwargs["data"]).encode("utf-8")
                headers["Content-Type"] = "application/x-www-form-urlencoded"
            request = urllib.request.Request(
                url, data=body, headers=headers, method=method
            )
            try:
                with opener.open(request, timeout=15) as response:
                    return Response(response.status, response.read())
            except urllib.error.HTTPError as error:
                return Response(error.code, error.read())
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(2 + attempt)
    raise RuntimeError(f"GitHub request failed after retries: {last_error}")


def wait_for_token(device_code: str) -> str:
    payload = {
        "client_id": CLIENT_ID,
        "device_code": device_code,
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
    }
    interval = 5
    deadline = time.time() + 880
    while time.time() < deadline:
        try:
            response = request_with_retry(
                "POST",
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json", "User-Agent": "GitHub CLI"},
                data=payload,
            )
        except RuntimeError:
            time.sleep(interval)
            continue
        result = response.json()
        if result.get("access_token"):
            return str(result["access_token"])
        error = result.get("error")
        if error == "slow_down":
            interval += 5
        elif error not in {"authorization_pending", None}:
            raise RuntimeError(f"GitHub authorization failed: {error}")
        time.sleep(interval)
    raise RuntimeError("GitHub device authorization timed out")


def request_device_code() -> str:
    response = request_with_retry(
        "POST",
        "https://github.com/login/device/code",
        headers={"Accept": "application/json", "User-Agent": "GitHub CLI"},
        data={"client_id": CLIENT_ID, "scope": "repo"},
    )
    response.raise_for_status()
    result = response.json()
    print(
        "GITHUB_DEVICE_AUTH "
        f"{result['verification_uri']} CODE {result['user_code']}",
        flush=True,
    )
    return str(result["device_code"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-code")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()

    device_code = args.device_code or request_device_code()
    token = wait_for_token(device_code)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Codex-Quant-Publisher",
    }
    user_response = request_with_retry(
        "GET", "https://api.github.com/user", headers=headers
    )
    user_response.raise_for_status()
    login = str(user_response.json()["login"])
    repository_url = f"https://github.com/{login}/{args.repo}"

    create_response = request_with_retry(
        "POST",
        "https://api.github.com/user/repos",
        headers=headers,
        json={
            "name": args.repo,
            "description": "Frozen 75/25 Alpha158-Barra LightGBM A-share research strategy",
            "private": False,
            "has_issues": True,
            "has_projects": False,
            "has_wiki": False,
            "auto_init": False,
        },
    )
    if create_response.status_code in {403, 422}:
        existing = request_with_retry(
            "GET", f"https://api.github.com/repos/{login}/{args.repo}", headers=headers
        )
        existing.raise_for_status()
        if existing.json().get("size", 0):
            raise RuntimeError("Target repository already exists and is not empty")
    else:
        create_response.raise_for_status()

    credential = (
        f"protocol=https\nhost=github.com\nusername={login}\npassword={token}\n\n"
    )
    subprocess.run(
        ["git", "credential", "approve"],
        input=credential,
        text=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    remote = f"https://github.com/{login}/{args.repo}.git"
    existing_remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=args.release,
        text=True,
        capture_output=True,
    )
    if existing_remote.returncode == 0:
        subprocess.run(
            ["git", "remote", "set-url", "origin", remote],
            cwd=args.release,
            check=True,
        )
    else:
        subprocess.run(
            ["git", "remote", "add", "origin", remote],
            cwd=args.release,
            check=True,
        )
    environment = os.environ.copy()
    subprocess.run(
        [
            "git",
            "-c",
            "http.sslBackend=openssl",
            "push",
            "-u",
            "origin",
            "main",
        ],
        cwd=args.release,
        env=environment,
        check=True,
    )
    print(f"PUBLISHED {repository_url}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"PUBLISH_FAILED {error}", file=sys.stderr)
        raise
