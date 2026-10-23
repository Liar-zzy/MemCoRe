"""Small, shared readers for service configuration (no client initialization)."""

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .config import ConfigError


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SECRET_KEYS = {"api_key", "password", "token", "secret", "authorization"}


def expand_environment(value: Any) -> Any:
    """Expand ${NAME} / ${NAME:-default} in parsed values, without editing YAML."""
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        name, default = match.groups()
        result = os.environ.get(name)
        if result:
            return result
        if default is not None:
            return default
        raise ConfigError(f"Required environment variable is not set: {name}")

    return _ENV.sub(replace, value)


def load_yaml(path, resolve_environment=True) -> dict:
    """Read a mapping and resolve service credentials from the environment."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except yaml.YAMLError:
        # Parser diagnostics can contain the original credential-bearing line.
        raise ConfigError(f"Invalid YAML in {path}") from None
    if not isinstance(value, dict):
        raise ConfigError(f"Expected a YAML mapping in {path}")
    return expand_environment(value) if resolve_environment else value


def redact(value: Any) -> Any:
    """Return a copy suitable for printing or recording in results."""
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if str(key).lower() in _SECRET_KEYS and item is not None
                  else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value
