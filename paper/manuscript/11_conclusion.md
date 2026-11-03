---
title: Conclusion
status: draft-3
---

# 8. Conclusion

We proposed an architecture for deep-search agents that moves retrieved documents out of the prompt and into the program, where the model reads them with code and a search cap bounds retrieval. In controlled comparisons it answered more questions than the standard function-calling agent, its in-prompt variants and Fetch-then-Explore, a recent workspace-based agent, also on questions held out from development, with about a fifth of the reference agent's tokens and a quarter of Fetch-then-Explore's per question, and ablations traced the gain to the wide searches the architecture makes affordable. Keeping documents outside the prompt is what makes wide searches affordable, and the REPL is what made the model use them: with the same store read through tools, GPT-5 kept its searches narrow and answered fewer questions, and even when made to search as wide it trailed by 4.7 to 8.0 points, a gap we could not resolve. It runs with an open-weight model, which makes distilling or fine-tuning its behaviour into smaller models a natural next step.
