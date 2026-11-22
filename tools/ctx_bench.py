"""usage: ctx_bench.py URL [--depths 1000,8000,32000,64000,100000] [--max 400] [--temp 0] [--label NAME]
Writing speed at different context depths. For each depth it sends a prompt holding about that many tokens of
Python source (from this machine's standard library) followed by a task, twice:
  1. a code task: the time to the first token is the time to read the context cold (prompt tokens per second);
  2. a prose task on the same context, which the server answers from its prefix cache.
For both it reports tokens per decoding pass, ms per pass and tokens per second while writing."""
import argparse, glob, json, os, sys, sysconfig, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("--depths", default="1000,8000,32000,64000,100000")
ap.add_argument("--max", type=int, default=400); ap.add_argument("--temp", type=float, default=0.0); ap.add_argument("--label", default=None)
ap.add_argument("--model", default="inkling-2-mini"); ap.add_argument("--effort", default="none")
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
corpus = "".join(f"# file: {os.path.basename(f)}\n" + open(f, errors="ignore").read() + "\n\n" for f in files)
TASKS = {"code": "Ignore the reference code above. Write a Python class implementing an LRU cache with get and put, with type hints, docstrings and a few unit tests.",
         "prose": "Ignore the reference code above. Explain how a bill becomes a law in the United States, in about 400 words."}
def ask(filler, task):
    content = (f"Reference code:\n```python\n{filler}\n```\n\n" if filler else "") + task
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": a.max, "temperature": a.temp, "stream": True,
            "stream_options": {"include_usage": True}, "logprobs": True, "top_logprobs": 0, "chat_template_kwargs": {"reasoning_effort": a.effort}}
    t0 = time.time(); first = None; passes = 0; toks = 0; usage = {}
    with urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}), timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:]); usage = d.get("usage") or usage
            ch = d.get("choices") or []
            n = len(((ch[0].get("logprobs") or {}).get("content") or [])) if ch else 0
            if n:
                first = first or time.time(); passes += 1; toks += n
    end = time.time(); dec = end - (first or end)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return {"prompt_tokens": usage.get("prompt_tokens"), "cached": cached, "ttft": (first or end) - t0, "tokens": toks, "passes": passes,
            "per_pass": toks / max(passes, 1), "ms_per_pass": 1000 * dec / max(passes - 1, 1), "tok_s": (toks - toks / max(passes, 1)) / dec if dec > 0 else 0}
ask("", TASKS["prose"])  # warm-up
out = []; chars_per_token = 3.0
print(f"{'context':>9s} {'read':>18s} | {'code: tok/s':>11s} {'per pass':>8s} {'ms/pass':>8s} | {'prose: tok/s':>12s} {'per pass':>8s} {'ms/pass':>8s}")
for depth in [int(x) for x in a.depths.split(",")]:
    # a unique first line, so no depth can reuse a shorter depth's prompt from the prefix cache: every read is cold
    filler = f"# context benchmark {depth} {time.time():.0f}\n" + corpus[: int(depth * chars_per_token)]
    if len(filler) < int(depth * chars_per_token) * 0.98:
        print(f"{depth:9d}  (not enough reference text on this machine)"); break
    c = ask(filler, TASKS["code"]); p = ask(filler, TASKS["prose"])
    new = c["prompt_tokens"] or 0
    chars_per_token = len(filler) / max((c["prompt_tokens"] or 1) - 60, 1)  # calibrate the next depth
    print(f"{c['prompt_tokens']:9d} {c['ttft']:6.1f} s {new / c['ttft']:7.0f} tok/s | {c['tok_s']:11.1f} {c['per_pass']:8.2f} {c['ms_per_pass']:8.1f} | {p['tok_s']:12.1f} {p['per_pass']:8.2f} {p['ms_per_pass']:8.1f}"
          + ("" if p["ttft"] < max(2.0, 0.2 * c["ttft"]) else f"   (prose request took {p['ttft']:.1f} s to start: not served from the cache)"), flush=True)
    out.append({"depth": depth, "code": c, "prose": p})
if a.label:
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(d, exist_ok=True)
    json.dump({"label": a.label, "temperature": a.temp, "max_tokens": a.max, "rows": out}, open(os.path.join(d, f"ctx-{a.label}.json"), "w"), indent=1)
