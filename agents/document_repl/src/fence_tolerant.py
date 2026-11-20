"""Opt-in parsing switches for models that mislabel code fences (used by the Qwen3.6 runs).

SUP_FENCE_TOLERANT=1: when a turn has no ```repl block, run other fenced blocks that parse as Python and do something
(assignment, call, import, loop, if), so an answer template in a bare fence is never executed.
SUP_MAX_BLOCKS=N: execute at most N code blocks per turn. Unset, both are no-ops.
"""
import ast
import os
import re

RESCUED = [0]


def _install_block_cap(RC) -> None:
    """SUP_MAX_BLOCKS=N: execute at most N code blocks per turn (bounds the output of repetition loops)."""
    cap = int(os.environ.get("SUP_MAX_BLOCKS", "0") or 0)
    if cap <= 0:
        return
    inner = RC.find_code_blocks

    def capped(text):
        blocks = inner(text)
        if len(blocks) > cap:
            print(f"[blockcap] {len(blocks)} code blocks in one turn -> executing the first {cap}",
                  flush=True)
            return blocks[:cap]
        return blocks

    RC.find_code_blocks = capped
    print(f"[blockcap] at most {cap} code blocks per turn", flush=True)


def install() -> bool:
    import rlm.core.rlm as RC  # the call site binds find_code_blocks in ITS namespace
    if os.environ.get("SUP_FENCE_TOLERANT", "") != "1":
        _install_block_cap(RC)
        return False

    orig = RC.find_code_blocks
    fence = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)\n```", re.DOTALL)
    does = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Call, ast.Import, ast.ImportFrom,
            ast.For, ast.While, ast.If)

    def tolerant(text):
        blocks = orig(text) if text else []
        if blocks or not text:
            return blocks
        out = []
        for lang, body in fence.findall(text):
            b = body.strip()
            if lang.lower() == "repl" or not b:
                continue
            try:
                tree = ast.parse(b)
            except SyntaxError:
                continue
            if any(isinstance(n, does) for n in ast.walk(tree)):
                out.append(b)
        if out:
            RESCUED[0] += 1
        return out

    RC.find_code_blocks = tolerant
    print("[fence] tolerant code-block parsing ON (```repl still preferred)", flush=True)
    _install_block_cap(RC)   # wraps the tolerant parser, so the cap counts rescued blocks too
    return True
