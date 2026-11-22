"""usage: req.py URL MAX_TOKENS [prose|code] -> one temperature-0 request, prints tokens and tok/s"""
import json, sys, time, urllib.request
url, n = sys.argv[1], int(sys.argv[2]); kind = sys.argv[3] if len(sys.argv) > 3 else "prose"
p = {"prose": "Explain how a bill becomes a law in the United States, in about 400 words.",
     "code": "Write a Python class implementing an LRU cache with get and put, with type hints, docstrings and a few unit tests."}[kind]
body = {"model": "inkling-2-mini", "messages": [{"role": "user", "content": p}], "max_tokens": n, "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "none"}}
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}), timeout=600))
dt = time.time() - t; c = r["usage"]["completion_tokens"]
print(f"{c} tokens in {dt:.2f} s = {c / dt:.1f} tok/s")
