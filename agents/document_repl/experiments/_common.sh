# Sourced by every launcher here: locates the repository, sources configs/paths.env (or $PATHS_ENV) and defines
# SRC / REPO / HOOKS_DIR.
_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "$_here" in */experiments/*) ALG=$(cd "$_here/../.." && pwd) ;; *) ALG=$(cd "$_here/.." && pwd) ;; esac
REPO=$(cd "$ALG/../.." && pwd)
SRC=$ALG/src
PATHS_ENV=${PATHS_ENV:-$REPO/configs/paths.env}
[ -f "$PATHS_ENV" ] || { echo "missing $PATHS_ENV (copy configs/paths.example.env and fill it in)" >&2; exit 2; }
set -a; . "$PATHS_ENV"; set +a
: "${RLMLIB_PATH:?set RLMLIB_PATH in configs/paths.env}" "${FULLDOC_ROOT:?set FULLDOC_ROOT in configs/paths.env}"
: "${RUNS_DIR:?set RUNS_DIR in configs/paths.env}" "${PYTHON:?set PYTHON in configs/paths.env}"
# Startup hooks (sitecustomize.py directories retry429/, run5d/, ...) belong to the Snippet RLM algorithm.
# HOOKS_DIR must point at them; it is checked only by the launchers that load a hook.
HOOKS_DIR=${HOOKS_DIR:-$REPO/agents/snippet_repl/src/hooks}
need_hook() { [ -f "$HOOKS_DIR/$1/sitecustomize.py" ] || { echo "hook $1 not found: set HOOKS_DIR (directory holding $1/sitecustomize.py)" >&2; exit 2; }; }
