"""Load Azure credentials for the agents and the judge without printing them.

Reads KEY=VALUE lines from .secrets/x3.env ($X3_ENV or --env-file) without overriding set variables, then fills the
canonical names from the aliases below.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = ROOT / ".secrets" / "x3.env"
ALIASES = {
    "AZURE_OPENAI_ENDPOINT": ["AZ_EP", "AZURE_ENDPOINT"],
    "AZURE_OPENAI_API_KEY": ["AZ_KEY"],
    "GPT5_DEPLOYMENT": ["X3_DEPLOYMENT", "AZURE_DEPLOYMENT_GPT5", "AZURE_OPENAI_DEPLOYMENT"],
    "JUDGE_DEPLOYMENT": ["X3_JUDGE_DEPLOYMENT", "AZURE_DEPLOYMENT_JUDGE", "AZURE_OPENAI_JUDGE_MODEL"],
}
NEEDED = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "GPT5_DEPLOYMENT", "JUDGE_DEPLOYMENT")


def load_env(path: str | None = None) -> Path | None:
    p = Path(path or os.environ.get("X3_ENV") or DEFAULT_ENV).expanduser()
    loaded = None
    if p.is_file():
        for line in p.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("export "):
                s = s[len("export "):]
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            k, v = k.strip(), v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            if k and k not in os.environ:
                os.environ[k] = v
        loaded = p
    for target, sources in ALIASES.items():
        if not os.environ.get(target):
            for s in sources:
                if os.environ.get(s):
                    os.environ[target] = os.environ[s]
                    break
    ep = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if ep:
        os.environ["AZURE_OPENAI_ENDPOINT"] = ep.strip().rstrip("/")
    # one deployment for everything: without a separate judge deployment, the GPT-5 deployment also judges
    if not os.environ.get("JUDGE_DEPLOYMENT") and os.environ.get("GPT5_DEPLOYMENT"):
        os.environ["JUDGE_DEPLOYMENT"] = os.environ["GPT5_DEPLOYMENT"]
        os.environ["JUDGE_IS_GPT5"] = "1"
    return loaded


def judge_note() -> str:
    """One line to print wherever the judge is used."""
    if os.environ.get("JUDGE_IS_GPT5"):
        return ("judge: the GPT-5 deployment (no JUDGE_DEPLOYMENT set). The archived rows were graded by gpt-4.1 with the same "
                "template; accuracies are then not judged alike, and a model grading its own answers may be lenient.")
    return f"judge: {os.environ.get('JUDGE_DEPLOYMENT')}"


def status() -> dict:
    """Which required variables are set (booleans only)."""
    return {k: bool(os.environ.get(k)) for k in NEEDED}


def judge_credentials() -> tuple[str, str]:
    """Endpoint and key for the judge; a separate resource may be given with X3_JUDGE_ENDPOINT / X3_JUDGE_API_KEY."""
    return (os.environ.get("X3_JUDGE_ENDPOINT") or os.environ["AZURE_OPENAI_ENDPOINT"],
            os.environ.get("X3_JUDGE_API_KEY") or os.environ["AZURE_OPENAI_API_KEY"])
