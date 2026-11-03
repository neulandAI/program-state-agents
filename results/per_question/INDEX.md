| file | configuration | questions | correct |
|---|---|---|---|
| x7_dense_gpt5.csv | reference agent | 150 | 104 |
| x7_dense_gpt5_rep2.csv | reference agent (run 2) | 150 | 97 |
| x7_dense_gpt5_getdoc.csv | + document tool | 150 | 118 |
| x7_dense_gpt5_getdoc_680.csv | + document tool, 680 held out | 680 | 527 |
| x7_dense_gpt5_getdoc_cap10.csv | + document tool, cap 10 | 150 | 105 |
| x7_dense_gpt5_getdoc_cap10_k25.csv | + document tool, cap 10, 25 snippets | 150 | 113 |
| x7_dense_gpt5_trim30k.csv | trimmed (30k) | 150 | 93 |
| x7_dense_gpt5_sum30k.csv | summarised at 30k | 150 | 107 |
| snip_dense_high150.csv | snippet REPL, no cap | 150 | 117 |
| snip_cap10.csv | snippet REPL, cap 10 | 150 | 109 |
| snip_k25.csv | snippet REPL, 25 snippets | 150 | 112 |
| snip_k25_rep2.csv | snippet REPL, 25 snippets (run 2) | 150 | 117 |
| fd_dense_high150.csv | document REPL, cap 10 | 150 | 125 |
| fd_rep2.csv | document REPL, cap 10 (run 2) | 150 | 130 |
| fd_nogrep.csv | document REPL, no grep tool | 150 | 124 |
| fd_k5.csv | document REPL, k = 5 | 150 | 105 |
| fd_k5_rep2.csv | document REPL, k = 5 (run 2) | 150 | 108 |
| fd_k5prompt.csv | document REPL, k = 5, prompt only | 150 | 107 |
| x8_toolstore.csv | tool store, cap 10 | 150 | 112 |
| x9_toolstore_k25.csv | tool store, cap 10, k = 25 | 150 | 118 |
| fd_cap20_150.csv | document REPL, cap 20 | 150 | 136 |
| fd_cap20_830.csv | document REPL, cap 20, all 830 | 830 | 724 |
| fd_colbert20_830.csv | document REPL, cap 20, late interaction, all 830 | 830 | 736 |
| x5_closed_book.csv | closed book | 150 | 28 |
| fte_full830.csv | Fetch-then-Explore (released harness, no cap), all 830 | 830 | 650 |
