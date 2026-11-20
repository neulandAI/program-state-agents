# Full-Doc RLM, budget 20, on all 830 BrowseComp-Plus questions · Reason-ModernColBERT, 8192-token PLAID index (rebuilt) · GPT-5 high · gpt-4.1 judge (3 votes)

| Right | Accuracy | Wilson 95 % | Searches / q (mean, median) | At the cap of 20 | Mean k | Gold recall | Every gold doc | Acc. with every gold | No gold doc | Acc. with none | Final prompt median / p90 / max (proxy) | Cost / q (list) | Wall time median / mean / max | Statuses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 736/830 | 0.8867 | 0.863–0.907 | 13.9, 16 | 40 % | 26.9 | 0.876 | 80 % | 0.950 | 7 % | 0.224 | 20,908 / 39,344 / 81,627 | 0.52 USD (total 432) | 356 / 467 / 5416 s | {'completed': 827, 'timeout': 3}, forced 0, timed out 0 |

Archived BCP-23 (same design and retriever, budget 10) on the same 830: mean searches 8.1, cost 0.58 USD/q.

## Paired against the archived full-set rows (exact McNemar; identical questions)

| Archived row | acc. budget 20 | acc. archived | only budget 20 | only archived | both | neither | p |
|---|---|---|---|---|---|---|---|
| Full-Doc RLM, dense, budget 10 (archived, 0.8313) | 0.8867 | 0.8313 | 88 | 42 | 648 | 52 | 6.7e-05 |
| Snippet RLM, dense, unlimited (archived, 0.794) | 0.8867 | 0.7940 | 122 | 45 | 614 | 49 | 2.1e-09 |
| Full-Doc RLM, ColBERT, budget 10 (archived, 0.8747) | 0.8867 | 0.8747 | 44 | 34 | 692 | 60 | 0.31 |
| Snippet RLM, ColBERT (archived, 0.8578) | 0.8867 | 0.8578 | 70 | 46 | 666 | 48 | 0.032 |
| Snippet RLM, BM25 (archived, 0.6747) | 0.8867 | 0.6747 | 210 | 34 | 526 | 60 | 3.8e-32 |
| Full-Doc RLM, BM25, budget 10 (archived, 0.600) | 0.8867 | 0.6000 | 261 | 23 | 475 | 71 | 2.9e-52 |

## Accuracy by archived difficulty (number of the six archived GPT-5 configurations that solved the question)

| Archived rows solving | n | Right | Accuracy |
|---|---|---|---|
| 0 | 31 | 7 | 0.226 |
| 1 | 21 | 10 | 0.476 |
| 2 | 42 | 28 | 0.667 |
| 3 | 63 | 48 | 0.762 |
| 4 | 153 | 137 | 0.895 |
| 5 | 181 | 173 | 0.956 |
| 6 | 339 | 333 | 0.982 |

Source: `scripts/full830_summary.py --run fd_colbert20_830`; records `data/x3/fd_colbert20_830/`; archived per-question verdicts `data/failure_analysis/bcp_by_question.csv`.
