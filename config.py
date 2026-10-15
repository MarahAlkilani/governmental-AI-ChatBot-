"""Configuration for the offline local Ollama integration."""

from __future__ import annotations

import os
from dataclasses import dataclass

LOCAL_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 45.0


class ConfigurationError(RuntimeError):
    """Raised when required local-generation configuration is absent or invalid."""


@dataclass(frozen=True)
class OllamaSettings:
    model: str
    host: str = LOCAL_OLLAMA_HOST
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


def get_ollama_settings() -> OllamaSettings:
    """Read explicit local Ollama settings without selecting a fallback model."""
    model = os.getenv("OLLAMA_MODEL", "").strip()
    if not model:
        raise ConfigurationError(
            "OLLAMA_MODEL is required; configure the locally installed Ollama model explicitly."
        )

    raw_timeout = os.getenv("OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError as exc:
        raise ConfigurationError(
            "OLLAMA_TIMEOUT_SECONDS must be a positive number of seconds."
        ) from exc

    if timeout_seconds <= 0:
        raise ConfigurationError("OLLAMA_TIMEOUT_SECONDS must be greater than zero.")

    return OllamaSettings(model=model, timeout_seconds=timeout_seconds)
