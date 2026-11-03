---
title: Results
status: draft-3
---

# 4. Results

**Table 1.** Every configuration on the same 150 questions (GPT-5, dense retrieval), grouped by agent. Columns as defined in Section IV; Reading (n): accuracy on the n questions with every gold document returned; snippet REPL tokens include the sub-model. Bold: best accuracy and reading.

| Configuration | Acc.% | Searches | Recall | Reading% (n) | Prompt | State | Tokens in / out |
|---|---|---|---|---|---|---|---|
| *Reference agent and in-prompt variants* | | | | | | | |
| Reference agent | 69.3 | 21.1 | 0.80 | 83 (107) | 63.5k | – | 885k / 10k |
| + document tool | 78.7 | 15.3 | 0.78 | 95 (104) | 48.3k | – | 696k / 9k |
| + document tool, cap 10 | 70.0 | 8.6 | 0.67 | **98** (85) | 35.7k | – | 255k / 9k |
| + document tool, cap 10, 25 snippets | 75.3 | 7.5 | 0.76 | 95 (100) | 127.9k | – | 672k / 7k |
| History trimmed at 30k | 62.0 | 35.1 | 0.77 | 79 (106) | 14.6k | – | 420k / 13k |
| History summarised at 30k | 71.3 | 26.6 | 0.82 | 85 (112) | 29.0k | – | 639k / 23k |
| *Snippet REPL agent* | | | | | | | |
| No cap | 78.0 | 25.7 | 0.79 | 91 (101) | 24.1k | 4k | 148k / 24k |
| Cap 10 | 72.7 | 8.3 | 0.62 | 95 (75) | 15.5k | 1k | 44k / 10k |
| No cap, 25 snippets | 74.7 | 24.7 | 0.88 | 83 (118) | 26.7k | 4k | 179k / 18k |
| *Fetch-then-Explore* | | | | | | | |
| No cap | 78.0 | 14.8 | 0.75 | 95 (97) | 35.1k | 3k | 610k / 39k |
| *Document REPL agent (ours)* | | | | | | | |
| Cap 10 | 83.3 | 8.1 | 0.78 | **98** (103) | 19.9k | 638k | 149k / 36k |
| Cap 20 | **90.7** | 13.4 | 0.88 | **98** (122) | 20.7k | 828k | 158k / 35k |
| *Ablations of ours (cap 10)* | | | | | | | |
| No grep tool | 82.7 | 8.0 | 0.76 | 97 (98) | 19.6k | 608k | 189k / 41k |
| *k* = 5 | 70.0 | 8.8 | 0.62 | 96 (80) | 21.1k | 108k | 153k / 43k |
| *k* = 5, documents in the prompt only | 71.3 | 7.7 | 0.64 | 95 (79) | 79.2k | – | 257k / 11k |
| Tool store, *k* chosen | 74.7 | 7.3 | 0.75 | 92 (93) | 23.6k | 237k | 235k / 8k |
| Tool store, *k* = 25 | 78.7 | 7.2 | 0.77 | 96 (103) | 24.9k | 596k | 254k / 9k |
| *No retrieval* | | | | | | | |
| Closed book | 18.7 | 0 | – | – | – | – | 0.2k / 15k |

**Table 2.** Our agent compared with each alternative on the same questions, exact McNemar test. Diff.: our accuracy minus the row's, in points. †: significant after Holm's correction (Section IV).

| Compared with | Acc.% | Diff. | *p* |
|---|---|---|---|
| *150 questions; ours, cap 10: 83.3%* | | | |
| Reference agent | 69.3 | +14.0 | < 0.001† |
| + document tool | 78.7 | +4.7 | 0.17 |
| + document tool, cap 10 | 70.0 | +13.3 | < 0.001† |
| + doc. tool, cap 10, 25 snippets | 75.3 | +8.0 | 0.043 |
| History summarised | 71.3 | +12.0 | 0.005 |
| Snippet REPL agent | 78.0 | +5.3 | 0.18 |
| *680 held-out questions; ours, cap 20: 86.6%* | | | |
| Reference agent + document tool | 77.5 | +9.1 | < 0.001 |
| Fetch-then-Explore | 78.4 | +8.2 | < 0.001 |
| *Ablations of ours (150 questions, cap 10)* | | | |
| *k* = 5 per search | 70.0 | +13.3 | < 0.001† |
| Tool store, *k* chosen | 74.7 | +8.7 | 0.015 |
| Tool store, *k* = 25 | 78.7 | +4.7 | 0.14 |
| No `grep_docs` | 82.7 | +0.7 | 1 |

## 4.1 RQ1: Effectiveness

Tables I and II compare the document REPL agent with the reference agent and its in-prompt variants on the same 150 questions. At a cap of 10 searches it answers 83% of the 150 questions versus 69% for the reference agent. The best in-prompt variant, the reference agent with the uncapped document tool, answers 79%, 4.7 points below ours, a difference that is not significant on 150 questions. Held to the same cap of 10, every in-prompt variant falls further behind, and trimming or summarising the history does not help. Our agent also uses far fewer tokens: 185k per question against the reference agent's 895k, about a fifth, with a final prompt of 20k against 64k tokens. Its input is a sixth of the reference agent's; its output, mostly code, is 3.7 times larger.

On the 680 questions held out from development, at the tuned cap of 20, it answers 86.6% versus 77.5% for the reference agent with the document tool; the gap stays significant even if 37 judge verdicts are flipped against our agent. Fetch-then-Explore [@liu2026fte], which also keeps fetched documents out of the prompt but reads them through tools, five search results at a time, answers 78.4% on the same questions. It finds less of the gold evidence (recall 0.76 against 0.87; on 119 of the 830 questions no search returned a gold document, against 39 for ours, Fig. 4) and uses about four times our tokens: 772k per question against 197k, with a median of 326k against 145k.

**Answer to RQ1.** The architecture scores higher than the reference agent and every in-prompt variant on the 150 development questions, significantly so against the reference agent and its capped variant, and on the 680 held-out questions it answers 87% versus 78% for both the reference agent with the document tool and Fetch-then-Explore, with about a quarter of Fetch-then-Explore's tokens. On the development questions it uses about a fifth of the reference agent's tokens.

## 4.2 RQ2: Mechanism

![Searching wider (150 questions): gold recall and accuracy of each agent at 5 documents or snippets per search (hollow) and at a wide setting (filled; the arrow runs from narrow to wide). Wide: 25 snippets per search for the capped reference agent with the document tool and for the snippet REPL agent, whose extra text enters the prompt; a mean of 28.8 documents chosen by the document REPL agent, whose extra text stays in program state. Square: an ablation of the document REPL agent at 5 per search with the documents printed into the prompt instead of held. Second runs: document REPL agent 72.0% at 5 per search and 86.7% wide; snippet REPL agent 78.0% wide.](paper/figures/fig3_mechanism.pdf){#fig:mech width=1col}

**Finding or reading.** The three main agents end with similar gold recall, 0.78 to 0.80 (for the snippet agents a returned snippet counts as found), but they get there differently: the reference agent through some twenty narrow searches and a growing prompt, ours through about eight wide searches and a short prompt. The difference shows in reading: on the questions where every gold document was returned, the reference agent answers 83%, the snippet REPL agent 91% and ours 98%, and on its longest prompts the reference agent's reading falls to 29% (n = 17). Length alone explains part of that: in the distractor test, up to 200k tokens of distractor text cost GPT-5 at most 11 points, significant only from 100k tokens on.

**Width.** Held to five documents per search (Fig. 3), our agent's gold recall falls from 0.78 to 0.62 and its accuracy from 83% to 70–72% (two runs), the reference agent's level. Its reading stays at 96%. Width matters for finding, not for reading, as Pi-Serini found for recall with deeper rankings [@hsu2026piserini]. Width placed in the prompt pays less (exploratory): with 25 snippets per search the capped reference agent gains five points from a 128k-token prompt, and the snippet REPL agent gains recall but not accuracy.

**Where the documents live.** With every retrieved document printed into the prompt and none held, at five per search, the agent answers as many questions as with the documents in program state (71.3 versus 70.0%, *p* = 0.84), but its median prompt grows from 21k to 79k tokens and exceeds the model's 272k-token input limit on 10 questions. At its usual width the agent holds a median 638k tokens, more than twice that limit.

**Code or tools.** The tool-store agent keeps our store but reads it through `read` and `grep` tools. Free to choose its width, it keeps its searches narrow, holding a median 41 documents versus our 105, and answers 75%. Made to take 25 documents per search, it finds as much as ours and answers 79%, 4.7 and 8.0 points behind our two runs, neither significant after correction; programmatic search agents likewise led tool agents on four of five models [@qian2026psa]. Only the document REPL agent chose to search wide on its own.

**The other components.** The `grep_docs` tool is not needed: without it the agent answers 83%, searching the held documents with Python instead. The cap sets how much the agent retrieves: raising it from 10 to 20 adds 11 questions on the 150 (*p* = 0.013) through recall (0.78 to 0.88), and 4 points on all 830 questions versus an earlier cap-10 run (Table III). History management is no substitute: deleting old results costs the reference agent 7 points (62.0 versus 69.3%, *p* = 0.019), and summarising them only recovers that loss (71.3%; *p* = 0.003 against trimming, significant after Holm's correction).

**Answer to RQ2.** The agents find about as much evidence but read it differently. Within the architecture, wide searches find the evidence, and keeping documents out of the prompt is what makes such searches affordable; at five documents per search, where the documents live made no detectable difference, and reading them in code rather than through tools added 4.7 to 8.0 points we could not resolve.

## 4.3 RQ3: Generality

**Table 3.** Accuracy (%) on all 830 BrowseComp-Plus and 824 FRAMES questions with GPT-5 and the open-weight Qwen3.6-35B-A3B. Snippet REPL agent uncapped on BrowseComp-Plus with GPT-5 and capped at 25 searches otherwise; cap 20 is the tuned setting. *p*: paired test of the snippet REPL and cap-10 document REPL columns. Published: the benchmark's GPT-5 reference agent [@chen2026browsecompplus]. Bold: best of the three agent columns.

| Model | Benchmark | Retriever | Snippet REPL | Document REPL, cap 10 | Document REPL, cap 20 | p | Published |
|---|---|---|---|---|---|---|---|
| GPT-5 | BrowseComp-Plus | dense | 79.4 | 83.1 | **87.2** | 0.012 | 70.1 |
| GPT-5 | BrowseComp-Plus | late interaction | 85.8 | 87.5 | **88.7** | 0.23 | – |
| GPT-5 | BrowseComp-Plus | BM25 | **67.5** | 60.0 | – | < 0.001 | 55.9 |
| GPT-5 | FRAMES | BM25 | 76.3 | **80.7** | – | < 0.001 | – |
| Qwen3.6 | BrowseComp-Plus | dense | – | **44.2** | – | – | – |
| Qwen3.6 | BrowseComp-Plus | late interaction | – | **58.0** | – | – | – |
| Qwen3.6 | BrowseComp-Plus | BM25 | **24.9** | 22.4 | – | – | – |
| Qwen3.6 | FRAMES | BM25 | **62.0** | 61.0 | – | – | – |

With GPT-5, the document REPL agent leads the snippet REPL agent on BrowseComp-Plus with dense retrieval and on FRAMES (Table III). With late interaction it also leads, though not significantly. With BM25 on BrowseComp-Plus it falls behind. Fig. 4 breaks down the remaining errors: with dense and late-interaction retrieval the document REPL agent has fewer wrong answers than the snippet REPL agent in every category; with BM25 most of its errors are questions where it found no gold document (171) or gave no answer (121), and it used all 30 turns on 169 questions.

![Failure modes on the full sets (GPT-5): every wrong answer as a share of the questions, for the snippet REPL agent (snippet), the document REPL agent (document; cap 20 with dense and late-interaction retrieval, cap 10 with BM25) and, with dense retrieval, Fetch-then-Explore. Nothing found: no gold document was ever returned; partly found: some but not all; all gold returned, still wrong: every gold document returned, inspected or not; no answer: unfinished or abstained.](paper/figures/fig4_failures.pdf){#fig:fail width=2col}

With the open-weight Qwen3.6 the architecture runs end to end and answers 44 and 58% with dense and late-interaction retrieval, against 83 and 87% with GPT-5 at the same cap, and with BM25 it scores slightly below the snippet REPL agent.

On OBLIQ-Bench (Table IV) the limit is first-stage retrieval, as its authors found [@tchuindjo2026obliqbench]: in our run logs either design's searches returned only 14 to 32% of the gold documents, and the benchmark's own agent leads on two of the three tasks. Wide searches cannot recover what the retriever never surfaces.

**Answer to RQ3.** With GPT-5 the architecture also leads on FRAMES and, not significantly, with late-interaction retrieval, and it runs with an open-weight model at lower accuracy. It does not help where the bottleneck lies elsewhere: in first-stage retrieval on OBLIQ-Bench and in BM25 retrieval on BrowseComp-Plus (Fig. 4).

**Table 4.** OBLIQ-Bench with GPT-5.2 and Gemini-Embedding-2: the two designs (20 results per search) and the benchmark's own agent (OBLIQ agent) [@tchuindjo2026obliqbench], on its gold judgements (×100). Bold: best per row.

| Task | Metric | Snippet REPL | Document REPL | OBLIQ agent |
|---|---|---|---|---|
| Math (151) | nDCG@10 | 14.9 | 15.9 | **16.1** |
| | Recall@10 | 14.1 | 13.6 | **14.5** |
| Twitter (281) | nDCG@10 | 10.2 | 9.5 | **14.1** |
| | Recall@10 | 10.6 | 9.5 | **14.1** |
| Congress (254) | nDCG@10 | **18.8** | 13.3 | 18.3 |
| | Recall@10 | **20.1** | 15.0 | 18.5 |
