# Tool-store agent (x9_toolstore_k25), 150 graded
accuracy 78.7 % (118/150); status {'completed': 150}
searches 7.2; reads 5.8; greps 2.3; model calls 15.4
gold recall 0.77; reading 96 % (n 103)
final prompt median 24,871 / p90 49,482 (API input tokens of the last call)
tokens in / out per question 254k / 9k

| vs | this / other correct | this only / other only | p |
|---|---|---|---|
| fd_dense_high150 | 118 / 125 | 5 / 12 | 0.1435 |
| fd_rep2 | 118 / 130 | 4 / 16 | 0.0118 |
| x7_dense_gpt5 | 118 / 104 | 22 / 8 | 0.0161 |
| x7_dense_gpt5_getdoc | 118 / 118 | 10 / 10 | 1.0000 |
| x7_dense_gpt5_getdoc_cap10 | 118 / 105 | 21 / 8 | 0.0241 |
| x7_dense_gpt5_getdoc_cap10_k25 | 118 / 113 | 14 / 9 | 0.4049 |
| snip_dense_high150 | 118 / 117 | 14 / 13 | 1.0000 |
| fd_k5 | 118 / 105 | 20 / 7 | 0.0192 |
