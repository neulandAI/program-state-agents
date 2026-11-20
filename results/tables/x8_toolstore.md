# Tool-store agent (x8_toolstore), 150 graded
accuracy 74.7 % (112/150); status {'completed': 150}
searches 7.3; reads 5.7; greps 1.8; model calls 14.8
gold recall 0.75; reading 92 % (n 93)
final prompt median 23,589 / p90 48,760 (API input tokens of the last call)
tokens in / out per question 235k / 8k

| vs | this / other correct | this only / other only | p |
|---|---|---|---|
| fd_dense_high150 | 112 / 125 | 6 / 19 | 0.0146 |
| fd_rep2 | 112 / 130 | 3 / 21 | 0.0003 |
| x7_dense_gpt5 | 112 / 104 | 23 / 15 | 0.2559 |
| x7_dense_gpt5_getdoc | 112 / 118 | 10 / 16 | 0.3269 |
| x7_dense_gpt5_getdoc_cap10 | 112 / 105 | 19 / 12 | 0.2810 |
| x7_dense_gpt5_getdoc_cap10_k25 | 112 / 113 | 13 / 14 | 1.0000 |
| snip_dense_high150 | 112 / 117 | 14 / 19 | 0.4869 |
| fd_k5 | 112 / 105 | 18 / 11 | 0.2649 |
