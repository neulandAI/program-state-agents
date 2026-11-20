#!/usr/bin/env bash
# Set or inspect the Azure credentials in .secrets/x3.env without ever printing a value.
#
#   evaluation/x3_secret.sh set KEY   # prompts with hidden input in a terminal, otherwise reads stdin
#   evaluation/x3_secret.sh status    # which keys are set, with the length of each value only
#   evaluation/x3_secret.sh unset KEY
#   evaluation/x3_secret.sh ping      # one tiny call per deployment: checks auth and the served model
#
# Keys: AZURE_OPENAI_ENDPOINT  AZURE_OPENAI_API_KEY  GPT5_DEPLOYMENT
# Optional: JUDGE_DEPLOYMENT (gpt-4.1; without it the GPT-5 deployment also judges)  AZURE_OPENAI_API_VERSION
#           X3_JUDGE_ENDPOINT  X3_JUDGE_API_KEY (only if the judge sits on another resource)
set -euo pipefail
shopt -s extglob
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="$ROOT/.secrets/x3.env"
ALLOWED="AZURE_OPENAI_ENDPOINT AZURE_OPENAI_API_KEY GPT5_DEPLOYMENT JUDGE_DEPLOYMENT AZURE_OPENAI_API_VERSION X3_JUDGE_ENDPOINT X3_JUDGE_API_KEY"
mkdir -p "$ROOT/.secrets"; chmod 700 "$ROOT/.secrets"
touch "$ENV"; chmod 600 "$ENV"

check_key() {
  case " $ALLOWED " in *" $1 "*) ;; *) echo "unknown key '$1'. Allowed: $ALLOWED" >&2; exit 2 ;; esac
}

cmd="${1:-status}"
case "$cmd" in
  set)
    key="${2:?usage: scripts/x3_secret.sh set KEY}"; check_key "$key"
    if [ -t 0 ]; then
      read -r -s -p "Value for $key (hidden): " val; echo
    else
      IFS= read -r val || true
    fi
    val="${val%$'\r'}"
    val="${val##+([[:space:]])}"; val="${val%%+([[:space:]])}"
    val="${val#\"}"; val="${val%\"}"; val="${val#\'}"; val="${val%\'}"
    val="${val#"$key"=}"
    if [ -z "$val" ]; then echo "empty value; nothing written" >&2; exit 1; fi
    case "$key" in *ENDPOINT) val="${val%/}" ;; esac
    tmp="$(mktemp "$ROOT/.secrets/.tmp.XXXXXX")"
    grep -v "^${key}=" "$ENV" > "$tmp" || true
    printf '%s=%s\n' "$key" "$val" >> "$tmp"
    mv "$tmp" "$ENV"; chmod 600 "$ENV"
    echo "$key set (${#val} characters)"
    ;;
  unset)
    key="${2:?usage: scripts/x3_secret.sh unset KEY}"; check_key "$key"
    tmp="$(mktemp "$ROOT/.secrets/.tmp.XXXXXX")"
    grep -v "^${key}=" "$ENV" > "$tmp" || true
    mv "$tmp" "$ENV"; chmod 600 "$ENV"
    echo "$key removed"
    ;;
  status)
    echo "$ENV"
    for k in $ALLOWED; do
      v="$(grep "^${k}=" "$ENV" | tail -1 | cut -d= -f2- || true)"
      if [ -n "$v" ]; then printf '  %-26s set (%d characters)\n' "$k" "${#v}"; else printf '  %-26s -\n' "$k"; fi
    done
    ;;
  ping)
    exec "$ROOT/.venv/bin/python" "$ROOT/evaluation/x3_plain_agent.py" --ping
    ;;
  *)
    echo "usage: scripts/x3_secret.sh set KEY | unset KEY | status | ping" >&2; exit 2 ;;
esac
