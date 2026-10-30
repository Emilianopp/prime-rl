"""Read the active Copilot CLI login without exporting or displaying its token."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


class CopilotCredentialError(RuntimeError):
    """An actionable credential-store error containing no credential values."""


def _checked_token(value: str) -> str:
    token = value.strip()
    if not token or not token.isascii() or any(ord(char) < 33 or ord(char) == 127 for char in token):
        raise CopilotCredentialError(
            "Copilot credential is empty or invalid; run copilot login or replace the environment token."
        )
    return token


def _read_config(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError):
        raise CopilotCredentialError("Cannot read Copilot config; check COPILOT_HOME and file permissions.") from None
    # CLI config is JSONC. Preserve quoted strings while removing comments and trailing commas.
    text = re.sub(
        r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/',
        lambda match: match[0] if match[0].startswith('"') else " ",
        text,
    )
    text = re.sub(
        r'"(?:\\.|[^"\\])*"|,\s*(?=[}\]])',
        lambda match: match[0] if match[0].startswith('"') else "",
        text,
    )
    try:
        config = json.loads(text)
    except ValueError:
        raise CopilotCredentialError(
            "Invalid Copilot config JSONC; repair the config or run copilot login again."
        ) from None
    if not isinstance(config, dict):
        raise CopilotCredentialError("Copilot config must be a JSON object.")
    return config


def _account(config: dict[str, Any]) -> tuple[str, str]:
    user = config.get("lastLoggedInUser")
    if not user:
        users = config.get("loggedInUsers") or []
        if isinstance(users, list) and len(users) == 1:
            user = users[0]
    if not isinstance(user, dict) or not all(isinstance(user.get(k), str) and user[k] for k in ("host", "login")):
        raise CopilotCredentialError(
            "No active Copilot CLI account. Run copilot login or select an account with /user."
        )
    host, login = user["host"], user["login"]
    if host not in ("github.com", "https://github.com"):
        raise CopilotCredentialError(
            "This CAPI judge requires a GitHub.com login; enterprise residency hosts are unsupported."
        )
    return host, login


def _keyring_token(account: str) -> str | None:
    try:
        import keyring
        from keyring.errors import KeyringError
    except ImportError:
        pass
    else:
        try:
            token = keyring.get_password("copilot-cli", account)
        except KeyringError:
            token = None
        if isinstance(token, str) and token.strip():
            return token.strip()

    command = None
    if sys.platform == "darwin" and (binary := shutil.which("security")):
        command = [binary, "find-generic-password", "-s", "copilot-cli", "-a", account, "-w"]
    elif sys.platform.startswith("linux") and (binary := shutil.which("secret-tool")):
        command = [binary, "lookup", "service", "copilot-cli", "account", account]
    if command is None:
        return None
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    try:
        return result.stdout.decode("utf-8").strip() or None
    except UnicodeError:
        raise CopilotCredentialError("OS keyring returned an invalid credential; run copilot login again.") from None


def github_token() -> str:
    """COPILOT_GITHUB_TOKEN overrides the active account's saved CLI login."""
    if "COPILOT_GITHUB_TOKEN" in os.environ:
        return _checked_token(os.environ["COPILOT_GITHUB_TOKEN"])

    path = Path(os.environ.get("COPILOT_HOME", str(Path.home() / ".copilot"))) / "config.json"
    config = _read_config(path)
    host, login = _account(config)
    account = f"{host}:{login}"
    if token := _keyring_token(account):
        return _checked_token(token)

    tokens = config.get("authTokens") or {}
    record = tokens.get(account) if isinstance(tokens, dict) else None
    token = record.get("token") if isinstance(record, dict) else None
    if isinstance(token, str) and token.strip():
        if os.name == "posix":
            try:
                info = path.stat()
            except OSError:
                raise CopilotCredentialError("Cannot verify Copilot credential-file permissions.") from None
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise CopilotCredentialError(
                    "Copilot credential config must be owned by you and private; set its permissions to 600."
                )
        return _checked_token(token)
    raise CopilotCredentialError(
        "Saved Copilot credential is unavailable. Unlock the OS keyring (Linux: install libsecret-tools), "
        "or run copilot login and choose an available credential store."
    )
