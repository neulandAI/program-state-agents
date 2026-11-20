# Holm over 18 contrasts (Table II, 150 questions)

| Contrast | n | first only / second only | p | Holm threshold | survives |
|---|---|---|---|---|---|
| ours vs + document tool, cap 10 | 150 | 24 / 4 | 0.00018 | 0.0028 | yes |
| k ~ 25 vs k = 5 (run 2) | 150 | 19 / 2 | 0.00022 | 0.0029 | yes |
| code vs tools, k chosen (run 2) | 150 | 21 / 3 | 0.00028 | 0.0031 | yes |
| k ~ 25 vs k = 5 (run 1) | 150 | 25 / 5 | 0.00032 | 0.0033 | yes |
| ours vs reference agent | 150 | 29 / 8 | 0.00075 | 0.0036 | yes |
| summarised vs trimmed | 150 | 17 / 3 | 0.0026 | 0.0038 | yes |
| ours vs summarised history | 150 | 28 / 10 | 0.0051 | 0.0042 | no |
| code vs tools, k = 25 (run 2) | 150 | 16 / 4 | 0.012 | 0.0045 | no |
| cap 20 vs cap 10 | 150 | 14 / 3 | 0.013 | 0.0050 | no |
| code vs tools, k chosen (run 1) | 150 | 19 / 6 | 0.015 | 0.0056 | no |
| untouched vs trimmed | 150 | 15 / 4 | 0.019 | 0.0063 | no |
| ours vs + document tool, cap 10, 25 snippets | 150 | 21 / 9 | 0.043 | 0.0071 | no |
| code vs tools, k = 25 (run 1) | 150 | 12 / 5 | 0.14 | 0.0083 | no |
| ours vs + document tool | 150 | 13 / 6 | 0.17 | 0.0100 | no |
| ours vs snippet REPL agent | 150 | 18 / 10 | 0.18 | 0.0125 | no |
| program state vs prompt only, k = 5 (run 1) | 150 | 11 / 13 | 0.84 | 0.0167 | no |
| program state vs prompt only, k = 5 (run 2) | 150 | 14 / 13 | 1 | 0.0250 | no |
| with vs without grep tool | 150 | 11 / 10 | 1 | 0.0500 | no |
