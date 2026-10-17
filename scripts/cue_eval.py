"""Live cue check: replay N trigger moments from held-out calls through /api/cue, as the page sends them, and measure
latency and output checks. GLM calls: N (default 30), sequential. Do not run during a demo or a panel run.

Moments: the page's own triggers (scripts/triggers.py) on the committed validation tags and flags, drawn with seed 7,
stratified by trigger kind. Checks per answer: a cue came back; `say` is present and in the call's language (stopword
test); `say` gives no dosing or medication instruction; the cue is at most 45 words.
Writes runs/cue-eval-<UTC stamp>.json with ids, kinds and numbers only (no transcript or cue text).

Usage: cue_eval.py <data dir> [n] [baseUrl]
"""
import json, os, random, re, statistics, sys, time, urllib.error, urllib.request
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triggers import play

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30
BASE = (sys.argv[3] if len(sys.argv) > 3 else os.environ.get("COACH_COPILOT_URL", "https://coach.raava.dev")).rstrip("/")
ES = re.compile(r"\b(que|de|el|la|los|las|y|en|con|para|por|tu|usted|esta|qué|cómo|lo)\b", re.I)
EN = re.compile(r"\b(the|and|you|your|to|of|is|that|what|how|it|this)\b", re.I)
DOSING = re.compile(r"\b(\d+\s?(mg|ml|pills?|tablets?)|take (an? |two |some )?(tylenol|ibuprofen|advil|aspirin|pill|dose)|tome (una|dos)? ?(pastilla|dosis)|dosis)\b", re.I)

lines = pd.read_csv(f"{DATA}/synthetic_data/synthetic_call_transcripts.csv")
calls = pd.read_csv(f"{DATA}/synthetic_data/synthetic_calls.csv").set_index("call_id")
P = os.path.join(ROOT, "runs", "predictions")
tags = pd.read_json(os.path.join(P, "tags-validation.jsonl"), lines=True)
flags = pd.read_json(os.path.join(P, "flags-validation.jsonl"), lines=True)
t = tags.merge(flags, on=["call_id", "line_number"], how="left").merge(lines[["call_id", "line_number", "speaker", "text"]], on=["call_id", "line_number"])
moments = []
for call_id, g in t.sort_values(["call_id", "line_number"]).groupby("call_id"):
    rows = [{"speaker": r.speaker, "text": r.text, "tag": r.pred if isinstance(r.pred, str) else None,
             "flag": r.flag if isinstance(r.flag, str) else "none", "n": r.line_number} for r in g.itertuples()]
    for i, kind, trigger in play(rows):
        moments.append({"call_id": call_id, "line": rows[i]["n"], "kind": kind, "trigger": trigger,
                        "recent": [{k: x[k] for k in ("speaker", "text", "tag")} for x in rows[max(0, i - 7):i + 1]]})
rng = random.Random(7)
by_kind = {}
for m in moments:
    by_kind.setdefault(m["kind"], []).append(m)
kinds = sorted(by_kind)
sample = []
while len(sample) < N and any(by_kind.values()):
    for k in kinds:
        if by_kind[k] and len(sample) < N:
            sample.append(by_kind[k].pop(rng.randrange(len(by_kind[k]))))

results = []
for m in sample:
    req = urllib.request.Request(BASE + "/api/cue", json.dumps({"trigger": m["trigger"], "recent": m["recent"]}).encode(),
                                 {"content-type": "application/json", "user-agent": "coach-copilot-cue-eval/1.0"})
    t0 = time.perf_counter()
    try:
        o, status = json.load(urllib.request.urlopen(req, timeout=90)), 200
    except urllib.error.HTTPError as err:
        o, status = {}, err.code
    wall = round((time.perf_counter() - t0) * 1000)
    lang = calls.loc[m["call_id"], "language"]
    say, cue = o.get("say") or "", o.get("cue") or ""
    es, en = len(ES.findall(say)), len(EN.findall(say))
    results.append({"call_id": m["call_id"], "line": m["line"], "kind": m["kind"], "language": lang, "status": status,
                    "cached": o.get("cached"), "workerMs": o.get("ms"), "roundTripMs": wall, "cue": bool(cue), "say": bool(say),
                    "sayLanguageOk": bool(say) and ((es > en) == (lang == "Spanish")), "noDosing": not DOSING.search(say + " " + cue),
                    "cueWords": len(cue.split())})
    time.sleep(1.5)

fresh = [r for r in results if r["status"] == 200 and not r["cached"]]
q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else None
share = lambda k: f"{sum(r[k] for r in results)}/{len(results)}"
summary = {"n": len(results), "answered": sum(r["status"] == 200 for r in results), "rateLimited": sum(r["status"] == 429 for r in results),
           "cueReturned": share("cue"), "sayReturned": share("say"), "sayInCallLanguage": share("sayLanguageOk"), "noDosing": share("noDosing"),
           "cueAtMost45Words": f"{sum(r['cue'] and r['cueWords'] <= 45 for r in results)}/{len(results)}",
           "latencyFresh": {"n": len(fresh), "workerMsP50": q([r["workerMs"] for r in fresh], .5), "workerMsP95": q([r["workerMs"] for r in fresh], .95),
                            "roundTripMsP50": q([r["roundTripMs"] for r in fresh], .5), "roundTripMsP95": q([r["roundTripMs"] for r in fresh], .95)},
           "byKind": {k: sum(r["kind"] == k for r in results) for k in kinds}, "momentsAvailable": len(moments)}
stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
path = os.path.join(ROOT, "runs", f"cue-eval-{stamp}.json")
json.dump({"ranAt": stamp, "base": BASE, "summary": summary, "results": results}, open(path, "w"), indent=1)
print(json.dumps(summary), "->", os.path.relpath(path, ROOT))
