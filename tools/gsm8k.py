"""Accuracy gate: score a model on GSM8K test questions through an OpenAI-compatible endpoint.

usage: gsm8k.py URL LABEL [--n 400] [--parallel 4] [--model NAME] [--effort low] [--pad TOKENS]
Temperature 0. The same first N questions every time, so runs are comparable question by question.
Writes results/LABEL.json (per-question correctness) and prints the accuracy.
With N = 400 and accuracy near 94%, one run's standard error is about 1.2 points: this catches a broken
kernel, not a half-point drift.
"""
import argparse, concurrent.futures as cf, json, os, random, re, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("label")
ap.add_argument("--n", type=int, default=400); ap.add_argument("--parallel", type=int, default=4)
ap.add_argument("--model", default="inkling-2-mini"); ap.add_argument("--effort", default="low")
ap.add_argument("--flush-on-burst", default=None, metavar="ADMIN_KEY", help="when 4 of the last 6 answers are wrong, call /flush_cache once ('-' for no key)")
ap.add_argument("--waves", action="store_true", help="send requests in simultaneous groups of --parallel, so the server prefills them as one batch")
ap.add_argument("--pad", type=int, default=0, help="prepend about this many tokens of unrelated text, unique per question")
a = ap.parse_args()
rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "gsm8k-test.jsonl"))][: a.n]


WORDS = ("river stone window garden letter morning silver market bridge candle forest harbor winter coffee paper mountain station yellow quiet travel number kitchen simple shadow pocket village music history rain").split()


def num(text):
    text = text.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(m[-1]) if m else None


TIMES = {}
FLUSHED = []


def flush(after):
    req = urllib.request.Request(a.url + "/flush_cache", b"", {} if a.flush_on_burst == "-" else {"Authorization": "Bearer " + a.flush_on_burst}, method="POST")
    try:
        r = urllib.request.urlopen(req, timeout=60).read()[:80]
    except Exception as e:
        r = repr(e)[:80]
    FLUSHED.append((after, time.time(), str(r)))
    print(f"  flush_cache after question {after}: {r}", flush=True)


def ask(i):
    t_start = time.time()
    q = rows[i]["question"]
    if a.pad:
        rng = random.Random(1000 + i)
        filler = " ".join(rng.choice(WORDS) for _ in range(int(a.pad * 0.8)))
        q = (f"Reference notes {i} (unrelated to the question, ignore them): {filler}.\n\nQuestion: " + q)
    body = {"model": a.model, "temperature": 0, "max_tokens": 3000, "chat_template_kwargs": {"reasoning_effort": a.effort},
            "messages": [{"role": "user", "content": q + "\n\nSolve this. Finish with a final line of the form: Answer: <number>"}]}
    for attempt in range(2):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(),
                                                 {"Content-Type": "application/json"}), timeout=900))
            break
        except Exception as e:
            if attempt: return i, False, None, 0, "error: " + repr(e)[:80]
    text = d["choices"][0]["message"].get("content") or ""
    TIMES[i] = (t_start, time.time(), d["usage"].get("prompt_tokens"), (d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"))
    tail = text.rsplit("Answer:", 1)[-1] if "Answer:" in text else text[-200:]
    got, gold = num(tail), num(rows[i]["answer"].rsplit("####", 1)[-1])
    return i, got is not None and abs(got - gold) < 1e-4, got, d["usage"]["completion_tokens"], d["choices"][0]["finish_reason"]


t0 = time.time(); res = {}; toks = 0
def results():
    if not a.waves:
        with cf.ThreadPoolExecutor(a.parallel) as ex:
            yield from ex.map(ask, range(len(rows)))
        return
    for w in range(0, len(rows), a.parallel):
        group = list(range(w, min(w + a.parallel, len(rows))))
        barrier = threading.Barrier(len(group))
        def go(i):
            barrier.wait()
            return ask(i)
        with cf.ThreadPoolExecutor(len(group)) as ex:
            yield from ex.map(go, group)


if True:
    for k, (i, ok, got, c, fin) in enumerate(results()):
        res[i] = {"ok": ok, "got": got, "tokens": c, "finish": fin}; toks += c
        if i in TIMES: res[i].update(t0=TIMES[i][0], t1=TIMES[i][1], prompt_tokens=TIMES[i][2], cached=TIMES[i][3])
        recent = [res[j]["ok"] for j in sorted(res)[-6:]]
        if a.flush_on_burst and not FLUSHED and len(recent) == 6 and recent.count(False) >= 4:
            flush(i)
        if (k + 1) % 100 == 0: print(f"  {k + 1} done, running accuracy {sum(r['ok'] for r in res.values()) / len(res) * 100:.1f}%", flush=True)
dt = time.time() - t0
acc = sum(r["ok"] for r in res.values()) / len(res)
errs = sum(1 for r in res.values() if str(r["finish"]).startswith("error"))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump({"label": a.label, "flushed": FLUSHED, "n": len(res), "pad": a.pad, "parallel": a.parallel, "accuracy": acc, "seconds": dt, "tokens": toks, "effort": a.effort, "results": res},
          open(os.path.join(HERE, "results", a.label + ".json"), "w"), indent=1)
print(f"{a.label}: {sum(r['ok'] for r in res.values())}/{len(res)} = {acc * 100:.1f}% correct, {toks} tokens in {dt:.0f} s ({toks / dt:.0f} tok/s in total), request errors: {errs}")
