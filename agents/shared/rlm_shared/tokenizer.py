"""Tokenizer used to cut BrowseComp-Plus snippets to a fixed number of tokens (Qwen/Qwen3-0.6B, as in the benchmark)."""
_TOKENIZER = None


def snippet_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return _TOKENIZER
