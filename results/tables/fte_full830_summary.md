# Fetch-then-Explore (full830) on BrowseComp-Plus, GPT-5 high, dense retrieval, gpt-4.1 judge

## Table I row (150 development questions)

| Acc.% | Searches | Recall | Reading% (n) | Prompt (median) | State (median) | Tokens in / out (mean) |
|---|---|---|---|---|---|---|
| 78.0 | 14.8 | 0.75 | 95 (97) | 35.1k | 3k | 610k / 39k |

## Paired comparison with the document REPL agent, cap 20 (exact McNemar)

| Questions | n | Fetch-then-Explore % | Ours % | only ours | only FtE | p | recall FtE / ours | tokens per question, mean (median), FtE / ours |
|---|---|---|---|---|---|---|---|---|
| all | 830 | 78.3 | 87.2 | 108 | 34 | 3.6e-10 | 0.76 / 0.87 | 750k (328k) / 196k (141k) |
| held out | 680 | 78.4 | 86.6 | 85 | 29 | 1.5e-07 | 0.76 / 0.87 | 772k (326k) / 197k (145k) |
| development | 150 | 78.0 | 90.0 | 23 | 5 | 0.00091 | 0.75 / 0.88 | 649k (330k) / 193k (128k) |
