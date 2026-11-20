# Full-Doc RLM, budget 20, on all 830 BrowseComp-Plus questions · dense (Qwen3-Embedding-8B, exact) · GPT-5 high · gpt-4.1 judge (3 votes)

| Right | Accuracy | Wilson 95 % | Searches / q (mean, median) | At the cap of 20 | Mean k | Gold recall | Every gold doc | Acc. with every gold | No gold doc | Acc. with none | Final prompt median / p90 / max (proxy) | Cost / q (list) | Wall time median / mean / max | Statuses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 724/830 | 0.8723 | 0.848–0.893 | 13.9, 17 | 44 % | 26.2 | 0.868 | 80 % | 0.952 | 8 % | 0.224 | 21,144 / 42,144 / 106,575 | 0.55 USD (total 455) | 280 / 391 / 5900 s | {'completed': 828, 'timeout': 2}, forced 0, timed out 0 |

Archived BCP-21 (same design and retriever, budget 10) on the same 830: mean searches 7.9, cost 0.44 USD/q.

## Paired against the archived full-set rows (exact McNemar; identical questions)

| Archived row | acc. budget 20 | acc. archived | only budget 20 | only archived | both | neither | p |
|---|---|---|---|---|---|---|---|
| Full-Doc RLM, dense, budget 10 (archived, 0.8313) | 0.8723 | 0.8313 | 58 | 24 | 666 | 82 | 0.00022 |
| Snippet RLM, dense, unlimited (archived, 0.794) | 0.8723 | 0.7940 | 101 | 36 | 623 | 70 | 2.5e-08 |
| Full-Doc RLM, ColBERT, budget 10 (archived, 0.8747) | 0.8723 | 0.8747 | 48 | 50 | 676 | 56 | 0.92 |
| Snippet RLM, ColBERT (archived, 0.8578) | 0.8723 | 0.8578 | 68 | 56 | 656 | 50 | 0.32 |
| Snippet RLM, BM25 (archived, 0.6747) | 0.8723 | 0.6747 | 197 | 33 | 527 | 73 | 1.2e-29 |
| Full-Doc RLM, BM25, budget 10 (archived, 0.600) | 0.8723 | 0.6000 | 253 | 27 | 471 | 79 | 3.4e-47 |

## Accuracy by archived difficulty (number of the six archived GPT-5 configurations that solved the question)

| Archived rows solving | n | Right | Accuracy |
|---|---|---|---|
| 0 | 31 | 1 | 0.032 |
| 1 | 21 | 8 | 0.381 |
| 2 | 42 | 23 | 0.548 |
| 3 | 63 | 45 | 0.714 |
| 4 | 153 | 139 | 0.908 |
| 5 | 181 | 174 | 0.961 |
| 6 | 339 | 334 | 0.985 |

## Judge consistency on the 150 records judged twice

The 150 subset records are the same files in `fd_cap20_150` and `fd_cap20_830`; judged twice (three votes each time), the verdict differs on 1 of 150 (q1214): accuracy 0.9067 first judging, 0.9000 second.

Source: `scripts/full830_summary.py --run fd_cap20_830`; records `data/x3/fd_cap20_830/`; archived per-question verdicts `data/failure_analysis/bcp_by_question.csv`.
