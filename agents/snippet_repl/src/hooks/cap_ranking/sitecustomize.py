"""Startup hook (OBLIQ_CAP_RANKING=1): at the turn cap, ask for the final ranking instead of a generic final answer."""
import os

if os.environ.get("OBLIQ_CAP_RANKING") == "1":
    try:
        import rlm.core.rlm as _R

        _MSG = ("You are out of turns: no more code will run. Reply now with your final ranking and nothing else: "
                "one docid per line, strongest to weakest, using the doc_ids of documents you retrieved. Put every "
                "plausible candidate you saw, best first. No code, no prose, no numbering.")
        _orig = _R.RLM._default_answer

        def _default_answer(self, message_history, lm_handler):
            hist = message_history + [{"role": "user", "content": _MSG}]
            response = lm_handler.completion(hist)
            if self.logger:
                self.logger.log(_R.RLMIteration(prompt=hist, response=response, final_answer=response, code_blocks=[]))
            return response

        _R.RLM._default_answer = _default_answer
    except Exception as e:  # noqa: BLE001
        print(f"[cap-ranking] hook not installed: {e}", flush=True)
