# Full-Doc RLM search-cap sweep · 50 stratified questions of the 150 subset · dense, GPT-5 high, gpt-4.1 judge

Cap 10 is the 150-question re-run restricted to the sample. Iteration cap 30 in every arm. Cost at list price, uncached.

| Cap | n | Accuracy | 95 % | Median searches | Mean | At search cap | At 30 iterations | Median prompt | Cost / q | vs cap 10: only this / only cap 10 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 50 | 0.760 (38/50) | 0.63–0.86 | 5 | 4.3 | 66% | 0% | 16,183 | 0.55 | 0 / 3 | 0.25 |
| 10 | 50 | 0.820 (41/50) | 0.69–0.90 | 10 | 8.1 | 50% | 0% | 17,996 | 0.49 | 0 / 0 | – |
| 15 | 50 | 0.800 (40/50) | 0.67–0.89 | 12 | 11.0 | 46% | 0% | 19,620 | 0.55 | 0 / 1 | 1 |
| 20 | 50 | 0.900 (45/50) | 0.79–0.96 | 16 | 13.9 | 42% | 0% | 20,766 | 0.54 | 5 / 1 | 0.22 |
| 25 | 50 | 0.840 (42/50) | 0.71–0.92 | 16 | 15.9 | 30% | 0% | 22,511 | 0.54 | 2 / 1 | 1 |
| 30 | 50 | 0.820 (41/50) | 0.69–0.90 | 20 | 19.3 | 38% | 0% | 24,323 | 0.58 | 2 / 2 | 1 |
