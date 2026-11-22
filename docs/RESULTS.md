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

## Speed by context depth

`tools/ctx_bench.py`: the prompt holds that many tokens of Python source followed by a task; temperature 0,
400 tokens written, lowest reasoning setting. "Read" is a cold read of the whole prompt (nothing cached). The
time per pass is the steady measure; tokens per second also moves with how many drafted tokens an answer
happens to accept (2.7 to 4.0 per pass on the code task here).

Default cache (pool of 117,888 tokens):

| Context | Cold read | Code task | Prose task | Time per pass, code / prose |
|---|---|---|---|---|
| 1K | 0.3 s | 87 tok/s | 64 tok/s | 32.3 / 29.5 ms |
| 9K | 2.4 s (3,800 tok/s) | 84 | 66 | 32.7 / 30.7 |
| 32K | 8.7 s (3,670 tok/s) | 91 | 58 | 35.3 / 31.2 |
| 64K | 19.4 s (3,330 tok/s) | 82 | 58 | 37.6 / 33.7 |
| 100K | 33.0 s (3,020 tok/s) | 85 | 55 | 40.0 / 36.0 |

8-bit cache (`KV=fp8`, pool of 235,776 tokens):

| Context | Cold read | Code task | Prose task | Time per pass, code / prose |
|---|---|---|---|---|
| 1K | 0.3 s | 86 tok/s | 59 tok/s | 33.1 / 29.8 ms |
| 36K | 11.2 s (3,190 tok/s) | 78 | 61 | 35.3 / 32.4 |
| 99K | 40.0 s (2,490 tok/s) | 70 | 54 | 38.4 / 35.7 |
| 149K | 70.1 s (2,120 tok/s) | 70 | 52 | 42.4 / 38.7 |
| 199K | 108.2 s (1,840 tok/s) | 87 | 50 | 45.8 / 41.9 |

A pass takes about 24% longer at 100K than at 1K, and about 38% longer at 199K. A follow-up request on the
same context starts in well under a second from the prefix cache, with one exception seen: at 100K on the
default cache (85% of the pool) the follow-up was read again in full.

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
| Writing at about 32K context | 91 code task, 58 prose | 52-62 | |
| Writing at about 50K context | 82 code task, 58 prose (64K) | 44-50 | 98-103 after a 53K prompt |
| Reading a long prompt | 3,620 tok/s at 29K | 291 at 32K | 2,590 at 53K |
| Weights | as published | attention, output layer and cache in 8-bit | re-quantised GGUF |

Sources: huggingface.co/sriharsha150/inkling-2-mini-vulkan, huggingface.co/Makdok/Inkling-2-Mini-AMD-GGUF.
Not run here; settings differ (temperature, prompt length), so treat the comparison as rough.
