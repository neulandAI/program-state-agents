"""Vertex AI client for Gemini embeddings.

The service-account JSON comes from NNEXT_PROVIDERS_GCP_VERTEX_SERVICE_ACCOUNT_JSON or, when unset, from the line of that
name in the .env-format file $GCP_VERTEX_ENV; it is never printed or stored. Location: $GEMINI_LOCATION (default "eu").
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "agents", "shared"))
from rlm_shared.paths import path  # noqa: E402

from google import genai
from google.genai import types
from google.oauth2 import service_account

VAR = "NNEXT_PROVIDERS_GCP_VERTEX_SERVICE_ACCOUNT_JSON"


def _sa_info():
    v = os.environ.get(VAR)
    if not v:
        for line in open(path("GCP_VERTEX_ENV")):
            if line.startswith(VAR + "="):
                v = line.split("=", 1)[1].strip()
                break
    if v[:1] in "'\"" and v[-1:] == v[:1]:
        v = v[1:-1]
    return json.loads(v)


def client(location=None):
    info = _sa_info()
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    # Without a per-request timeout a single stuck call can stall an index build indefinitely.
    return genai.Client(vertexai=True, project=info["project_id"],
                        location=location or os.environ.get("GEMINI_LOCATION", "eu"), credentials=creds,
                        http_options=types.HttpOptions(timeout=120_000))
