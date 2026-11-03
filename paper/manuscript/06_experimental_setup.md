---
title: Experimental setup
status: draft-3
---

# 3. Experimental Setup

**Benchmarks.** BrowseComp-Plus [@chen2026browsecompplus] fixes a corpus of 100,195 web pages for 830 hard multi-constraint questions from BrowseComp [@wei2025browsecomp] and labels, for every question, the *gold documents* that contain the answer. These labels let us tell an agent that never found the evidence from one that found it and misread it, which makes BrowseComp-Plus our controlled testbed. FRAMES [@krishna2025frames] (824 multi-hop questions over Wikipedia) and three ranking tasks of OBLIQ-Bench [@tchuindjo2026obliqbench], whose queries describe a latent property with little word overlap, test how far the architecture carries.

**Models.** GPT-5 [@openai2025gpt5systemcard] serves every call, including the snippet REPL agent's sub-model, at high reasoning effort in every 150-question run, the held-out reference run and both cap-20 full-set runs; the other full-set runs used the setting recorded with each run. To test the architecture without a frontier model, we also run the open-weight Qwen3.6-35B-A3B [@qwen2026qwen36]. On OBLIQ-Bench we use GPT-5.2 [@openai2025gpt52], the model of the benchmark's own agent.

**Retrieval.** Every agent on the 150 questions uses exact dense (nearest-neighbour) search over the benchmark's Qwen3-Embedding-8B embeddings [@zhang2025qwen3embedding]. The full-set runs add BM25, a lexical retriever [@robertson2009bm25], and Reason-ModernColBERT, a late-interaction retriever [@chaffin2025reasonmoderncolbert]; OBLIQ-Bench uses Gemini-Embedding-2 [@choi2026geminiembedding2].

**Development and held-out questions.** The controlled comparisons and ablations use 150 questions drawn at random, stratified by how often six earlier full-set runs of the two REPL designs had solved each. They are the development set for the REPL designs; the reference agent was run as published. The other 680 questions were not used for tuning, and the main comparison is repeated on them.

**Metrics.** *Accuracy*, judged by GPT-4.1 with the benchmark's template, majority of three votes; an unfinished question counts as wrong. *Gold recall*: the share of gold documents any search returned, read or not (for snippet agents a returned snippet counts). *Reading*: accuracy on the questions where all gold documents were returned. *Prompt*: tokens of the last model input. *State*: tokens held outside the prompt. *Tokens*: summed over a question's model calls.

**Statistics.** Agents are compared on the same questions with the exact McNemar test [@mcnemar1947]; Table II compares our agent with each alternative; comparisons not involving our agent are reported in the text. Repeated runs of the same configuration differ by up to seven questions (about five points), and a single accuracy on 150 questions has a 95% interval of about ±7 points, so we rely on paired tests. Holm's correction [@holm1979] over 18 comparisons on the 150 questions (the ten in Table II, four in Section V-B and four second runs) keeps those marked †; the others are exploratory. For the held-out comparison we also report how many judge verdicts would have to flip against our agent before it loses significance.

**Distractor test.** To separate prompt length from question difficulty, GPT-5 answers each of the 150 questions in one call given every gold document (each cut to 12k tokens, 40k in total), alone and with 25k to 200k tokens of the retriever's highest-ranked non-evidence pages added.

**Other runs.** The Table III runs (all 830 BrowseComp-Plus questions and FRAMES) use each agent's own configuration; their caps are not matched. The cap of 20 was chosen on 50 of the 150 questions. Table I's cap-20 row is the 150-question slice of the full-set run, judged separately; the two judgements differ on one question (90.7 versus 90.0%). The document REPL agent's cap-10 full-set runs used an earlier version of its harness with one more tool, `get_document`. We ran Fetch-then-Explore [@liu2026fte] with its released harness and defaults on all 830 questions: GPT-5, the same dense retriever, five 512-token snippets per search as in its harness's BrowseComp-Plus setting, no search cap, a 300-turn limit, and network access blocked except for the model endpoint. Prompts, harnesses, scripts and per-question outcomes are in our repository, without benchmark text.
