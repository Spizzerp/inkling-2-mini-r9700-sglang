"""Harder long-context check: several numbered facts are scattered through a long prompt and the question needs
four of them at once (their sum). The same seeds give the same prompts on every run, so two server
configurations can be compared prompt by prompt.
usage: multineedle.py URL TOKENS TRIALS [MODEL]"""
import json, random, sys, time, urllib.request
url, n, trials = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
model = sys.argv[4] if len(sys.argv) > 4 else "inkling-2-mini"
W = "river stone window garden letter morning silver market bridge candle forest harbor winter coffee paper mountain station yellow quiet travel".split()
NAMES = "amber birch cedar delta ember flint grove heron indigo juniper kestrel linden".split()
ok = 0; t_all = 0; toks = 0; detail = []
for trial in range(trials):
    rng = random.Random(5000 + trial * 97 + n)
    per = 14
    lines = [f"Entry {i}: " + " ".join(rng.choice(W) for _ in range(per - 2)) + "." for i in range(int(n / 1.19) // per)]
    values = {name: rng.randint(100, 999) for name in NAMES}
    for j, name in enumerate(NAMES):
        pos = int(len(lines) * (0.04 + 0.92 * j / (len(NAMES) - 1)))
        lines.insert(pos + j, f"Entry special: the stock count of the {name} warehouse is {values[name]}.")
    asked = rng.sample(NAMES, 4)
    want = sum(values[a] for a in asked)
    q = (f"Inventory log {trial}\n\n" + "\n".join(lines) + "\n\nAdd up the stock counts of these four warehouses: "
         + ", ".join(asked) + ". Finish with a final line of the form: Answer: <number>")
    body = {"model": model, "max_tokens": 2500, "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "low"},
            "messages": [{"role": "user", "content": q}]}
    t = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
    dt = time.time() - t; t_all += dt; toks = d["usage"]["prompt_tokens"]
    text = (d["choices"][0]["message"].get("content") or "")
    tail = text.rsplit("Answer:", 1)[-1] if "Answer:" in text else text[-80:]
    digits = "".join(c for c in tail.replace(",", "") if c.isdigit() or c == " ").split()
    got = int(digits[-1]) if digits else None
    good = got == want; ok += good; detail.append("." if good else "x")
print(f"{toks} prompt tokens, sum of 4 of 12 scattered facts: {ok}/{trials} correct [{''.join(detail)}], {t_all / trials:.1f} s per prompt")
