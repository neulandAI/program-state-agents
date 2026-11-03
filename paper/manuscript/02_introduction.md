---
title: Introduction
status: draft-3
---

# 1. Introduction

Deep-search agents answer a question by searching a corpus repeatedly and reasoning over what the searches return [@huang2025deepresearchagents]. A hard question combines several constraints whose evidence is spread over several documents, so a single retrieval step, as in standard retrieval-augmented generation, rarely suffices: on FRAMES, one BM25 retrieval lifts Gemini 1.5 Pro only from 41 to 47%, and multi-step retrieval to 66% [@krishna2025frames]. The agent needs many searches and many documents before it can answer. In the prevailing design, from the reason-and-act loop [@yao2023react], every search result is appended to the conversation [@chen2026browsecompplus]. This forces a trade-off. Short snippets keep the prompt small but hide evidence deeper in a page; whole documents expose the evidence, but the prompt grows with every search, and models read long prompts worse even when the needed text is present [@hong2025contextrot; @liu2024lostmiddle].

Recursive language models (RLMs) [@zhang2025rlm] show a way out of this trade-off for long inputs. An RLM never places its input in the prompt: the input is a variable in a Python read-eval-print loop (REPL), the model inspects it by writing code, can call itself recursively on parts of it, and sees only what its code prints. An RLM can be given a search tool, and we build one as a baseline, but the natural way to do so keeps the standard design's output: each search prints its snippets into the prompt, and whole documents are read through recursive sub-calls. The prompt still grows with every search, and reading a document costs further model calls.

We propose an architecture that brings the RLM idea to retrieval by moving retrieved text out of the prompt and into the program (Fig. 1).[^repo] The agent runs in the RLM's REPL, without recursive calls. Each search returns whole documents into a variable that persists across turns; the model reads them by writing code, for example slicing a document or matching a pattern across every held document, and only what its code prints enters the prompt. A search cap, a hard limit on searches per question, replaces the context window as the bound on how much the agent retrieves.

![Where the results of one search go, for each agent of Section III (purple: our architecture). Top of each card: what one search returns. Grey box: the prompt, which the model reads at every call. Dashed box: program state, which the model reaches only by running code; only what the code prints enters the prompt. Bottom: tokens one search adds to the prompt. The reference agent's document-tool variant also puts whole documents into the prompt.](paper/figures/fig1_architecture.pdf){#fig:arch width=2col}

Because a held document costs the prompt nothing until part of it is printed, the agent can afford to search wide: it chooses how many documents each search returns, up to 50, and in our runs asked for about 30. Wide searches find evidence that narrow ones miss, and selective printing keeps the prompt short enough to read well. The architecture builds on RLMs, on code as the action space [@wang2024codeact] and on tool results kept in the execution environment [@jones2025codemcp]; our contribution is its design for deep search over a retrieved corpus and the evidence for why it works.

We compare it with the standard function-calling agent and with variants of it that tackle the same trade-off inside the prompt: a whole-document tool to expose hidden evidence, a trimmed or summarised history to curb prompt growth, and the same search cap or wider searches to match the search budget. We pin the model, GPT-5 [@openai2025gpt5systemcard], not because the architecture depends on it but so that every agent uses the same model and every ablation changes exactly one thing. BrowseComp-Plus [@chen2026browsecompplus] is our controlled testbed, because its gold labels separate finding the evidence from reading it. FRAMES [@krishna2025frames] and OBLIQ-Bench [@tchuindjo2026obliqbench] test how far the architecture carries. The open-weight Qwen3.6 [@qwen2026qwen36] tests whether it works without a frontier model, a first step towards distilling or fine-tuning the behaviour into smaller models.

We ask three questions:

- **RQ1 (effectiveness).** Does the architecture answer more questions than the standard design and its in-prompt variants?
- **RQ2 (mechanism).** Why does it work: does it find more evidence or read it better, and which component is responsible?
- **RQ3 (generality).** Does it carry over to other benchmarks, retrievers and an open-weight model?

Our contributions are: (1) a program-state architecture for deep-search agents that keeps retrieved documents out of the prompt and bounds retrieval with a search cap, answering more questions than the standard design with fewer tokens than any of its in-prompt variants; (2) a controlled evaluation against the standard design and its in-prompt variants, including questions held out from development, with ablations that isolate search width, where documents live, reading in code and the `grep_docs` tool; (3) design guidance for agent engineers.

[^repo]: <https://github.com/neulandAI/program-state-agents>
