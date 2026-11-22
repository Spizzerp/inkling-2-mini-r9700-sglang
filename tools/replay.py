"""usage: replay.py URL --pad N --ids 236,237,238,239 [--together] [--max 3000] [--effort low] [--save DIR]
Re-send chosen gate questions exactly as gsm8k.py builds them: one at a time, or --together as one simultaneous wave.
Prints finish reason, tokens, the answer found and a hash of the text for each."""
import argparse, concurrent.futures as cf, hashlib, json, os, random, re, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("--pad", type=int, default=0); ap.add_argument("--ids", required=True)
ap.add_argument("--together", action="store_true"); ap.add_argument("--max", type=int, default=3000); ap.add_argument("--effort", default="low")
ap.add_argument("--model", default="inkling-2-mini"); ap.add_argument("--save", default=None)
a = ap.parse_args()
rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "gsm8k-test.jsonl"))]
src = open(os.path.join(HERE, "gsm8k.py")).read()
WORDS = eval(re.search(r"^WORDS = (\(.*\))", src, re.M).group(1))
def num(text):
    m = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "").replace("$", "")); return float(m[-1]) if m else None
def ask(i):
    q = rows[i]["question"]
    if a.pad:
        rng = random.Random(1000 + i); filler = " ".join(rng.choice(WORDS) for _ in range(int(a.pad * 0.8)))
        q = (f"Reference notes {i} (unrelated to the question, ignore them): {filler}.\n\nQuestion: " + q)
    body = {"model": a.model, "temperature": 0, "max_tokens": a.max, "chat_template_kwargs": {"reasoning_effort": a.effort},
            "messages": [{"role": "user", "content": q + "\n\nSolve this. Finish with a final line of the form: Answer: <number>"}]}
    d = json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    m = d["choices"][0]["message"]; text = m.get("content") or ""; rc = m.get("reasoning_content") or ""
    tail = text.rsplit("Answer:", 1)[-1] if "Answer:" in text else text[-200:]
    got, gold = num(tail), num(rows[i]["answer"].rsplit("####", 1)[-1])
    if a.save:
        os.makedirs(a.save, exist_ok=True); open(os.path.join(a.save, f"{i}.txt"), "w").write("REASONING:\n" + rc + "\n\nCONTENT:\n" + text)
    return i, d["choices"][0]["finish_reason"], d["usage"]["completion_tokens"], got, gold, hashlib.sha1((rc + text).encode()).hexdigest()[:10], len(rc)
ids = [int(x) for x in a.ids.split(",")]
if a.together:
    with cf.ThreadPoolExecutor(len(ids)) as ex: out = list(ex.map(ask, ids))
else:
    out = [ask(i) for i in ids]
for i, fin, tok, got, gold, h, rc in out:
    print(f"  q{i}: finish {fin:6s} {tok:5d} tok  got {got} gold {gold}  {'OK' if got is not None and abs(got - gold) < 1e-4 else 'WRONG'}  reasoning chars {rc}  text {h}")
