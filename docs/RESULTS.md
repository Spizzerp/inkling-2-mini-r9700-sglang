# Results and how they were measured

Machine: Radeon AI PRO R9700 32 GB (gfx1201), Intel Core i9-13900K, display on the integrated graphics,
Linux 7.2, rootless Docker. One request at a time unless stated. 22 November 2026, image of release 0.1.0,
except the sections on 0.1.1 and 0.1.2 and the speed by context depth (0.1.2).

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

## Start-up and the 0.1.1 check

Time from `docker run` to the first `GET /health` answering 200:

| Build | Seconds (separate starts) |
|---|---|
| 0.1.0 | 105.5, 98.3, 99.2; 100.4 with the 8-bit cache |
| + three kernel modules compiled into the image | 63.6, 63.6, 63.1, 64.0 |
| + Rust toolchain in the image | 57.6, 57.8 |
| + one warm-up pass per graph (0.1.1) | 47.5, 47.5, 47.0, 47.5; 47.7 with the 8-bit cache |

The table is from the development image, whose patched files are identical to this release's. The release
build started with `run.sh` (23 November): 177 s for the first start with an empty kernel cache
(`CACHE_DIR`), then 47.5, 47.5 and 47.0 s; 48.0 and 48.0 s with the 8-bit cache; 57.8 and 57.8 s with
`WARMUPS=2`. Two other starts, taken straight after the accuracy gate, took 4 s longer (52.1 s with the 8-bit
cache, 61.0 s with `WARMUPS=2`). On that build: an image request answered correctly, GSM8K 388/400 and
234/240 padded by 250 tokens with no request errors, and the fixed prompts of the Speed section at 104.8
tok/s (code) and 64.6 (prose).

0.1.1 changes nothing that runs once the server is up, apart from how often each graph ran before it was
recorded. It was compared with the 0.1.0 build in alternating runs at temperature 0, where this benchmark
repeats to within about 0.2% (six fixed prompts of 512 tokens, two of them code; "four at once" is the
total):

| | 0.1.0 (2 runs) | 0.1.1 (4 runs; code from 2) |
|---|---|---|
| Code prompts | 100.0, 99.8 tok/s | 99.6, 99.8 |
| Prose prompts | 69.7, 69.5 | 69.4 to 69.5 |
| Four requests at once | 139.7, 139.6 | 139.4 to 139.6 |
| Reading 3.3K tokens | 3,933, 3,933 tok/s | 3,920 to 3,926 |
| Reading 25.6K tokens | 3,767, 3,765 | 3,753 to 3,767 |

Reading came out 0.3% lower on 0.1.1 in three of the four runs, with one warm-up pass and with two; the cause
was not found. In two 0.1.1 runs the first code prompt overlapped the server's own warm-up request (the
benchmark had waited for `/v1/models` instead of `/health`), so their code figures are left out.

Accuracy gate on 0.1.1: 388/400 (the same as 0.1.0), 236/240 padded, 234/240 padded with the 8-bit cache, no
request errors. Four prompts with facts to recall at 72K tokens gave the same answers on both builds; single
facts at 105K (default cache) and 181K tokens (8-bit cache) were recalled correctly.

## 0.1.2: the draft shortlist

Release build `sglang-inkling-r9700:0.1.2` started with `run.sh`, 23 November. Its SGLang source files are
identical to those of the development image the 91-prompt figures below were measured on.

Fixed prompts (`tools/accept_ab.py`, temperature 0), the same image with `SHORTLIST=0` and with the list:

| | `SHORTLIST=0` | 0.1.2 |
|---|---|---|
| Code prompts | 104.0 tok/s | 118.3 tok/s |
| Prose | 64.2 | 74.5 |
| Tokens per pass, code / prose | 3.39 / 1.84 | 3.53 / 1.96 |
| Time per pass, code / prose | 32.6 / 28.7 ms | 29.8 / 26.3 ms |

With `SHORTLIST=0` all six answers are the same text as on 0.1.1. With the list the texts differ, so the
tokens per pass are not those of the same answers; the time per pass is the like-for-like figure. At
temperature 1, three runs: code 99.8, 112.5 and 107.7 tok/s, prose 69.5, 68.1 and 67.5 (29.1 to 29.9 and
25.8 to 26.1 ms per pass).

A wider set: 91 prompts that share no task with the prompts the list was built from, temperature 0, up to
600 tokens each, one request at a time:

| Kind (prompts) | Tokens per pass | Time per pass | 0.1.1 | 0.1.2 | Change |
|---|---|---|---|---|---|
| Code in 14 languages (40) | 2.90 to 2.88 | 31.7 to 29.2 ms | 91.2 tok/s | 98.7 | +8.2% |
| Tool calls (12) | 4.66 to 4.70 | 33.5 to 31.4 | 139.0 | 149.7 | +7.7% |
| Rewrites of a source file (12) | 3.63 to 3.51 | 33.5 to 31.0 | 108.2 | 113.0 | +4.4% |
| Config and data files (8) | 3.12 to 3.07 | 32.7 to 30.1 | 95.2 | 101.7 | +6.8% |
| Prose (12) | 2.10 to 2.07 | 29.6 to 26.9 | 70.8 | 76.9 | +8.6% |
| Prose in seven other languages (7) | 1.49 to 1.43 | 27.1 to 24.5 | 54.9 | 58.5 | +6.5% |
| All | 2.73 to 2.68 | 31.1 to 28.5 | 87.8 | 94.1 | +7.2% |

- Per answer the code change is +7.8% +/- 1.4. 13 of the 91 answers are token for token the same as on
  0.1.1; the others drift to different wording once a pass ends elsewhere, and 8 of the 40 code answers came
  out slower. Two unrestricted runs give 91 identical answers; draft mode off against on gives none of 16.
- By language (one prompt each): Spanish +8%, German +8%, French +4%, Russian +11%, Hebrew +16%,
  Japanese -4%, Chinese -11%.
- A benchmark with six fixed prompts of 512 tokens (two of them code), alternating runs: prose 69.4, 69.5
  against 74.3, 74.3 tok/s; four requests at once 139.4, 139.5 against 149.0, 149.1; code 99.5, 99.7 against
  95.6, 95.7. The code figure fell because the first of its two code answers changed its wording to one with
  twice as many one-token passes; none of the tokens the draft got confirmed before is off the list.
- Time to the first token is unchanged or 5 ms shorter at six prompt sizes from 75 to 11,000 tokens
  (medians of 7). Start-up 47.2, 47.1 and 48.0 s, 49.1 s with the 8-bit cache; the first start with an empty
  kernel cache 184.5 s. Context pool
  117,888 and 235,776 tokens as before. Peak VRAM in the benchmark 0.08 GiB higher.

Accuracy gate and checks on the release build: GSM8K 390/400, 235/240 padded by 250 tokens, 389/400 at
temperature 1, 234/240 padded with the 8-bit cache, no request errors; four prompts with scattered facts at
86K tokens 4/4; single facts at 105K (default cache) and 181K tokens (8-bit cache) recalled; an image
request answered correctly; `tools/test_draft_shortlist.py` passes.

## Speed by context depth

`tools/ctx_bench.py`: the prompt holds that many tokens of Python source followed by a task; temperature 0,
400 tokens written, lowest reasoning setting. "Read" is a cold read of the whole prompt (nothing cached). The
time per pass is the steady measure; tokens per second also moves with how many drafted tokens an answer
happens to accept (2.6 to 4.0 per pass on the code task here). Measured on 0.1.2; on 0.1.0 a pass took
32.3 / 29.5 ms at 1K and 40.0 / 36.0 ms at 100K on the default cache, 33.1 / 29.8 ms at 1K and 45.8 / 41.9 ms
at 199K on the 8-bit cache.

Default cache (pool of 117,888 tokens):

| Context | Cold read | Code task | Prose task | Time per pass, code / prose |
|---|---|---|---|---|
| 1K | 0.3 s | 89 tok/s | 70 tok/s | 29.6 / 27.0 ms |
| 9K | 2.4 s (3,760 tok/s) | 102 | 68 | 30.9 / 27.4 |
| 32K | 8.8 s (3,650 tok/s) | 115 | 67 | 34.2 / 29.5 |
| 64K | 19.5 s (3,310 tok/s) | 87 | 60 | 34.0 / 30.6 |
| 100K | 33.1 s (3,010 tok/s) | 88 | 59 | 37.1 / 33.6 |

8-bit cache (`KV=fp8`, pool of 235,776 tokens):

| Context | Cold read | Code task | Prose task | Time per pass, code / prose |
|---|---|---|---|---|
| 1K | 0.3 s | 88 tok/s | 66 tok/s | 30.6 / 27.5 ms |
| 36K | 11.3 s (3,170 tok/s) | 103 | 67 | 33.7 / 29.9 |
| 90K | 35.0 s (2,560 tok/s) | 107 | 58 | 37.4 / 32.6 |
| 133K | 59.8 s (2,220 tok/s) | 75 | 53 | 37.8 / 34.4 |
| 181K | 94.0 s (1,930 tok/s) | 69 | 52 | 40.5 / 37.5 |

A pass takes about 25% longer at 100K than at 1K, and about 32% longer at 181K. A follow-up request on the
same context starts in well under a second from the prefix cache, with two exceptions in these runs: at 100K
on the default cache (85% of the pool, also seen on 0.1.0) and at 133K on the 8-bit cache the follow-up was
read again in full.

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
Not run here; settings differ (temperature, prompt length), so treat the comparison as rough. The column
"This" holds the 0.1.0 figures; 0.1.2 writes about 7% faster.
