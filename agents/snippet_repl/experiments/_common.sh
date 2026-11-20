# Sourced by every launcher here: loads <repo>/configs/paths.env and defines ALG, SRC and snip_pythonpath.
# PYTHONPATH order: [hook dir under src/hooks] : rlm checkout (alexzhang13/rlm @ 854e688) : agents/shared (rlm_shared)
# The hook must come first: Python imports only the first sitecustomize.py.
ART=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ALG=$ART/agents/snippet_repl
SRC=$ALG/src
[ -f "$ART/configs/paths.env" ] || { echo "ABORT: $ART/configs/paths.env missing (copy configs/paths.example.env and fill it in)"; exit 2; }
set -a; . "$ART/configs/paths.env"; set +a
: "${RLMLIB_PATH:?set RLMLIB_PATH in configs/paths.env (rlm checkout at 854e688)}"
: "${PYTHON:?set PYTHON in configs/paths.env}"
snip_pythonpath() {  # $1 = hook directory name under src/hooks, or "" for none
  local p="${RLMLIB_PATH}:$ART/agents/shared"
  [ -n "$1" ] && p="$SRC/hooks/$1:$p"
  echo "$p"
}
