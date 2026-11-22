#!/usr/bin/env bash
# Downloads the GSM8K test set (MIT licence, github.com/openai/grade-school-math) for tools/gsm8k.py and tools/replay.py.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p data
curl -fsSL https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl -o data/gsm8k-test.jsonl
wc -l data/gsm8k-test.jsonl
