"""usage: accept_ab.py URL LABEL [TEMPERATURE] [MAX_TOKENS] -> per prompt and in total: tokens per decoding pass
(from streamed logprobs), ms per pass, tok/s, split into reasoning and answer, plus a hash of the text so two
temperature-0 runs can be checked for identical output. Saves JSON next to this file in results/."""
import collections, hashlib, json, os, sys, time, urllib.request
url, label = sys.argv[1], sys.argv[2]
temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 700
prompts = [("code", "Write a Python class implementing an LRU cache with get and put, with type hints, docstrings and a few unit tests."),
           ("code", "Write a Go function that merges two sorted slices of ints, with a table-driven test."),
           ("code", "Write a TypeScript function that debounces another function, with generics and a Jest test file."),
           ("code", "Write a Rust function that parses a simple INI file into a HashMap of sections, with error handling and tests."),
           ("prose", "Explain how a bill becomes a law in the United States, in about 400 words."),
           ("prose", "Write a short essay on why the Roman Empire declined.")]
out = {"label": label, "temperature": temp, "runs": []}
tot = {k: {"hist": collections.Counter(), "secs": 0.0} for k in ("code", "prose")}
for kind, p in prompts:
    body = {"model": "inkling-2-mini", "messages": [{"role": "user", "content": p}], "max_tokens": max_tokens, "temperature": temp, "stream": True,
            "logprobs": True, "top_logprobs": 0, "chat_template_kwargs": {"reasoning_effort": "none"}}
    hist = {"reasoning": collections.Counter(), "answer": collections.Counter()}; text = []
    t = time.time(); first = None
    with urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}), timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            ch = json.loads(line[6:]).get("choices") or []
            if not ch: continue
            d = ch[0].get("delta") or {}
            n = len((ch[0].get("logprobs") or {}).get("content") or [])
            phase = "reasoning" if d.get("reasoning_content") else "answer"
            text.append((d.get("reasoning_content") or "") + (d.get("content") or ""))
            if n:
                first = first or time.time(); hist[phase][n] += 1
    secs = time.time() - (first or t)
    allh = hist["reasoning"] + hist["answer"]
    toks = sum(n * c for n, c in allh.items()); passes = sum(allh.values())
    def mean(h): return sum(n * c for n, c in h.items()) / max(sum(h.values()), 1)
    digest = hashlib.sha1("".join(text).encode()).hexdigest()[:10]
    print(f"{kind:5s} {toks:4d} tok {passes:4d} passes  {toks / max(passes, 1):.2f}/pass  {1000 * secs / max(passes, 1):5.1f} ms/pass  {toks / secs:6.1f} tok/s"
          f"  reasoning {sum(n * c for n, c in hist['reasoning'].items()):4d} tok at {mean(hist['reasoning']):.2f}, answer at {mean(hist['answer']):.2f}  text {digest}", flush=True)
    tot[kind]["hist"] += allh; tot[kind]["secs"] += secs
    out["runs"].append({"kind": kind, "prompt": p, "tokens": toks, "passes": passes, "secs": secs, "text_sha1": digest,
                        "hist": {ph: dict(h) for ph, h in hist.items()}})
for kind, v in tot.items():
    h = v["hist"]; toks = sum(n * c for n, c in h.items()); passes = sum(h.values())
    print(f"== {label} {kind}: {toks / passes:.2f} per pass, {1000 * v['secs'] / passes:.1f} ms per pass, {toks / v['secs']:.1f} tok/s; "
          + " ".join(f"{n}:{c / passes * 100:.0f}%" for n, c in sorted(h.items())))
    out[kind] = {"per_pass": toks / passes, "ms_per_pass": 1000 * v["secs"] / passes, "tok_s": toks / v["secs"], "hist": dict(h)}
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"ab-{label}.json"), "w"), indent=1)
