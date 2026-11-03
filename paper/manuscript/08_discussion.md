---
title: Discussion
status: draft-3
---

# 5. Discussion

**Where it does not help.** When retrieval is the bottleneck, as on OBLIQ-Bench, searching wide over a weak ranking adds little; with BM25 on BrowseComp-Plus the agent often ran out of turns or gave no answer (Section V-C); and a smaller open-weight model runs the architecture but trails the frontier model by 20 to 40 points.

**Design guidance.**

- *Keep retrieved documents out of the prompt* and return one line per search. Read them with code or, where model-written code cannot be isolated, through `read` and `grep` tools, at a cost of 4.7 to 8.0 points we could not resolve.
- *Let the model search wide*, and fix the width if it does not choose to; bound retrieval with a search cap rather than the context window.
- *Choose the cap on a sample of questions* and check it on others.
- *If text must leave the prompt for good, summarise rather than delete.*

**Limitations.** Against the standard design our comparisons are between whole agents, which also differ in harness and prompt. The controlled comparisons use one model and 150 development questions, mostly in single runs; only the main comparison is repeated on held-out questions. Wide searches with whole documents in the prompt were not run, since they exceed the input limit; the location result therefore holds only at five documents per search. The capped reference variants learned the cap only when refused and had no initial search, and our snippet baselines show each document's first 512 tokens, without reranking [@sharifymoghaddam2026rerank]. We did not optimise how the agent searches: it writes its own queries and chooses its width with a hand-written prompt, and about half of its remaining errors are questions where it never retrieved all gold documents (Fig. 4). Natural next steps are systematic prompt optimisation such as GEPA [@agrawal2026gepa] and training the search policy on the agent's explicit trajectories. Accuracy comes from a large language model (LLM) judge, and the model may have seen the corpus: without tools it answers 19% of the 150 questions.
