---
title: Threats to validity
status: removed
---

# 6. Threats to validity

We follow the categories of Wohlin et al. [@wohlin2012experimentation].

**Construct.** Accuracy is an LLM judgement (gpt-4.1, three votes), the benchmark's own, with no human validation. "Every gold document retrieved" counts documents a search returned, not read; for the reference agent that is a 512-token snippet, and the cap-10 full-set active-observation runs logged evidence-document recall, so recall is not compared across Table 3. Delegated-observation token counts are lower bounds.

**Internal.** Every configuration is a single run; two runs of the same design and retriever disagree on 19-22 of 150 questions with a net difference of 0-1, so a difference of a few questions is not evidence, and the REPL remainder at cap ten and the late-interaction cap-20 difference are not significant. Cap 20 was selected on the 150 questions and confirmed on the 680 others. Fetch-then-Explore ran once with its released defaults, which were designed for open-web search and not tuned for GPT-5 or this corpus.

**External.** The main line is one benchmark, one model and one retriever; the full-set runs add two retrievers, two benchmarks and an open-weight model, and show the advantage of wide, capped searching depending on both; the Qwen3.6 rows count unfinished runs as wrong (up to 243 of 830), as do two reference-agent content refusals. The corpus is public web text that GPT-5 has seen in training: without any retrieval it answers 28 of the 150 questions (Table 2). Absolute accuracies will not transfer to a private corpus; the paired differences between agents are the portable result.

**Conclusion.** The central results (the main comparison, the evidence-conditioned split, k = 5, the distractor test at 200k, the dense full-set comparisons) have p below 0.001; the trimming, equal-cap and cap-20-on-150 contrasts do not survive a family-wise correction; on 150 questions a single accuracy has a 95 % interval of about ±7 points.
