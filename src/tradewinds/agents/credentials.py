"""Read the project-local Z.ai credential without evaluating shell syntax."""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def zai_api_key():
    """Prefer an explicit environment value over the ignored project .env file."""
    if "ZAI_API_KEY" in os.environ:
        return os.environ["ZAI_API_KEY"].strip()
    if not ENV_FILE.is_file():
        return ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == "ZAI_API_KEY":
            return value.strip().strip("\"'")
    return ""
