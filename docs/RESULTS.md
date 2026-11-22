# Results and how they were measured

Machine: Radeon AI PRO R9700 32 GB (gfx1201), Intel Core i9-13900K, display on the integrated graphics,
Linux 7.2, rootless Docker. One request at a time unless stated. 22 November 2026, image of release 0.1.0.

## Speed

Fixed prompts (`tools/accept_ab.py`: four code and two prose prompts, 700 tokens each, lowest reasoning
setting). "Before" is the same image with `ADAPT=0`.

| | Before | 0.1.0 |
|---|---|---|
| Code prompts, temperature 0 | 98.5 tok/s | 104.5 tok/s |
| Prose, temperature 0 | 57.1 | 64.4 |
| Code prompts, temperature 1 | 104.4 | 107.1 |
| Prose, temperature 1 | 58.1 | 63.2 |
| Tokens per pass, code / prose (temperature 0) | 3.31 / 1.89 | 3.39 / 1.84 |
| Time per pass, code / prose | 33.6 / 33.2 ms | 32.4 / 28.6 ms |

The code prompts include the model's reasoning, which is prose-like; pure code output accepts 4 to 7 tokens
per pass and runs faster than these figures.

A longer benchmark at temperature 1 (medians; single runs scatter by up to 20%): prose 70 tok/s, code 120,
a 5,000-token file rewrite 148, tool-call turns 200-209.

Draft mode off (`DRAFT=0`): 61 tok/s for everything, 16.4 ms per token.
Reading: a 29K-token prompt in 8 s (3,620 tok/s); 3,217 tok/s with the 8-bit cache; a repeated prompt in
0.12 s. A 120K-token prompt with one fact to recall: answered correctly in 53 s (8-bit cache).

## Accuracy gate

GSM8K test set, the first N questions, temperature 0, reasoning effort low (`tools/gsm8k.py`). With N = 400
one run's standard error is about 1.2 points: this catches a broken kernel, not a half-point drift.
"Padded" prepends unrelated text to every question; "waves" sends four at the same instant.

| Run | Default cache | 8-bit cache (`KV=fp8`) |
|---|---|---|
| 400 questions, four at once | 388 (97.0%) | |
| 200 / 120 questions, one at a time | 196 (98.0%) | 116 of 120 (96.7%) |
| 240 questions padded by 250 tokens, waves of four | 232 (96.7%) | 233 (97.1%) |
| 120 questions padded by 1,100 tokens, one at a time | 117 (97.5%) | |
| Draft mode off, 400 questions (reference, earlier build) | 389 (97.2%) | |

No request errors and no server errors in any run.

An answer that runs to the 3,000-token limit was the marker of the bugs fixed by patches 5, 6 and 8 (the
text was garbage). One answer in these 1,320 did so: question 236, padded, 8-bit cache, an ambiguous puzzle
about refilling pens. Replayed (`tools/replay.py`), the model deliberates coherently on it for 2,849, 1,639
and 613 tokens with patch 10 on and 1,183, 841 and 844 with it off, and answers correctly every time.
Question-by-question answer lengths against the previous build are balanced (3-4 answers more than twice as
long, 3-4 less than half), the same scatter as between two earlier builds.

## Against other published AMD results (their authors' figures, read on 22 November 2026)

| | This | Vulkan engine, R9700 | Patched llama.cpp, RX 7900 XTX |
|---|---|---|---|
| Code with draft mode | 112-125 tok/s | 118 (Python), 89 (C) | |
| General writing | 63-70 | 90-91 without draft mode | 98-103 with, 78 without |
| Reading a long prompt | 3,620 tok/s at 29K | 291 at 32K | 2,590 at 53K |
| Weights | as published | attention, output layer and cache in 8-bit | re-quantised GGUF |

Sources: huggingface.co/sriharsha150/inkling-2-mini-vulkan, huggingface.co/Makdok/Inkling-2-Mini-AMD-GGUF.
Not run here; settings differ (temperature, prompt length), so treat the comparison as rough.
