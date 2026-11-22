"""Long-context check: a unique prompt of about TOKENS tokens with one fact hidden in it.
usage: needle.py URL TOKENS [POSITION 0..1] [MODEL]"""
import json, os, random, sys, time, urllib.request
url, n = sys.argv[1], int(sys.argv[2])
pos = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
model = sys.argv[4] if len(sys.argv) > 4 else "inkling-2-mini"
rng = random.Random(n * 31 + int(pos * 1000) + int(time.time() * 1000) + os.getpid())
W = "river stone window garden letter morning silver market bridge candle forest harbor winter coffee paper mountain station yellow quiet travel".split()
code = rng.randint(100000, 999999)
per = 14
lines = [f"Entry {i}: " + " ".join(rng.choice(W) for _ in range(per - 2)) + "." for i in range(int(n / 1.19) // per)]
lines.insert(int(len(lines) * pos), f"Entry special: the access code for the blue vault is {code}.")
body = {"model": model, "max_tokens": 1500, "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "low"},
        "messages": [{"role": "user", "content": f"Log {rng.random()}\n\n" + "\n".join(lines) + "\n\nWhat is the access code for the blue vault? Answer with the number only."}]}
t = time.time()
try:
    d = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
except urllib.error.HTTPError as e:
    print(f"target {n}: HTTP {e.code} {e.read()[:160]!r}"); sys.exit(1)
dt = time.time() - t; u = d["usage"]; ans = (d["choices"][0]["message"].get("content") or "").strip()
print(f"prompt {u['prompt_tokens']} tokens, fact at {int(pos * 100)}%: {dt:.1f} s ({u['prompt_tokens'] / dt:.0f} tok/s incl. answer, {u['completion_tokens']} out, finish {d['choices'][0]['finish_reason']}), "
      f"{'CORRECT' if str(code) in ans else 'WRONG: ' + repr(ans[:30]) + ' expected ' + str(code)}")
