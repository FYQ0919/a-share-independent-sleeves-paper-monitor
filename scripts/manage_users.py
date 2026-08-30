from __future__ import annotations

import argparse
from getpass import getpass

from app.config import settings
from app.local_auth import AccountStore, _normalize_username


def _store() -> AccountStore:
    return AccountStore(
        settings.auth_database_path,
        settings.auth_lockout_attempts,
        settings.auth_lockout_minutes,
    )


def _password_twice() -> str:
    password = getpass("Password: ")
    if password != getpass("Confirm password: "):
        raise SystemExit("Passwords do not match")
    return password


def _require_user(changed: bool, username: str) -> None:
    if not changed:
        raise SystemExit(f"User not found: {username}")


def run(args) -> None:
    store = _store()
    if args.command == "list":
        print("username\tdisplay_name\tactive\tadmin\tlast_login")
        for user in store.list_users():
            print(
                f"{user['username']}\t{user['display_name']}\t"
                f"{bool(user['is_active'])}\t{bool(user['is_admin'])}\t"
                f"{user['last_login_at'] or '-'}"
            )
        return
    if args.command == "create":
        username = _normalize_username(args.username)
        password = _password_twice()
        errors = store.validate_registration(
            username,
            args.display_name,
            password,
            settings.auth_password_min_length,
        )
        if errors:
            raise SystemExit("; ".join(errors))
        user = store.create_user(username, args.display_name, password)
        if not user:
            raise SystemExit(f"Username already exists: {username}")
        if args.admin:
            store.set_admin(username, True)
        print(f"Created user: {username}")
        return
    if args.command == "reset-password":
        password = _password_twice()
        errors = store.validate_registration(
            _normalize_username(args.username),
            "account",
            password,
            settings.auth_password_min_length,
        )
        if errors:
            raise SystemExit("; ".join(errors))
        _require_user(store.reset_password(args.username, password), args.username)
        print(f"Password reset: {args.username}")
        return
    if args.command in {"enable", "disable"}:
        _require_user(store.set_active(args.username, args.command == "enable"), args.username)
        print(f"User {args.command}d: {args.username}")
        return
    if args.command in {"grant-admin", "revoke-admin"}:
        _require_user(
            store.set_admin(args.username, args.command == "grant-admin"),
            args.username,
        )
        print(f"Role updated: {args.username}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Manage local website accounts")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    create = commands.add_parser("create")
    create.add_argument("--username", required=True)
    create.add_argument("--display-name", required=True)
    create.add_argument("--admin", action="store_true")
    for name in (
        "reset-password",
        "enable",
        "disable",
        "grant-admin",
        "revoke-admin",
    ):
        command = commands.add_parser(name)
        command.add_argument("--username", required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
