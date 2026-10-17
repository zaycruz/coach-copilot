"""Second-layer model bake-off on dev lines where Clef was unsure (confidence < 0.3).

Usage: glm_bench.py <data_dir> [n_lines]
Prints, per model: review accuracy vs the expert label, Clef's accuracy on the same lines, and latency;
then each model's cue for a few triggered moments, for a side-by-side read.
"""
import json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

BASE = os.environ.get("COACH_COPILOT_URL", "https://coach.raava.dev").rstrip("/")
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["@cf/zai-org/glm-5.3", "@cf/zai-org/glm-5.3-flash", "@cf/zai-org/glm-4.7-flash"]


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"content-type": "application/json", "user-agent": "coach-copilot/0.1"})
    t = time.time()
    try:
        return json.load(urllib.request.urlopen(req, timeout=90)), time.time() - t
    except Exception as err:
        return {"error": str(err)}, time.time() - t


def main():
    data = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    lines = pd.read_csv(f"{data}/synthetic_data/synthetic_call_transcripts.csv")
    by_call = {cid: g.sort_values("line_number").to_dict("records") for cid, g in lines.groupby("call_id")}
    rows = [json.loads(l) for l in open(os.path.join(HERE, "..", "runs", "dev-clef-ctx2.jsonl"))]
    low = [r for r in rows if r["confidence"] is not None and r["confidence"] < 0.3]
    random.seed(11)
    sample = random.sample(low, min(n, len(low)))
    for r in sample:
        call = by_call[r["call_id"]]
        i = next(k for k, x in enumerate(call) if x["line_number"] == r["line_number"])
        r["previous"] = [{"speaker": x["speaker"], "text": x["text"]} for x in call[max(0, i - 2):i]]
    clef_acc = sum(r["pred"] == r["gold"] for r in sample) / len(sample)
    print(f"{len(sample)} low-confidence dev lines; Clef alone gets {clef_acc:.0%} of them")

    for model in MODELS:
        def review(r):
            return post("/api/review", {"model": model, "line": {"speaker": r["speaker"], "text": r["text"]},
                                         "previous": r["previous"], "probabilities": r["probabilities"]})
        with ThreadPoolExecutor(4) as pool:
            res = list(pool.map(review, sample))
        ok = [(out, secs) for out, secs in res if out.get("label")]
        acc = sum(out["label"] == r["gold"] for (out, _), r in zip(res, sample) if out.get("label")) / max(len(ok), 1)
        lat = sorted(secs for _, secs in ok) or [0]
        errs = [out.get("error") or "no label" for out, _ in res if not out.get("label")]
        if errs: print("   unanswered:", sorted(set(e[:90] for e in errs)))
        print(f"{model:30s} review acc {acc:.0%}  answered {len(ok)}/{len(sample)}  p50 {lat[len(lat)//2]:.1f}s  p90 {lat[int(len(lat)*.9)]:.1f}s")

    moments = [
        ("The coach just confronted the participant.", [("participant", "I forgot after Tuesday.", "sustain"), ("coach", "That sounds like an excuse to me.", "confront")]),
        ("The coach just told the participant what to do without asking permission.", [("participant", "Dinner is always late at our house.", "sustain"), ("coach", "You need to eat by 7, no excuses.", "persuade")]),
        ("The participant voiced change talk and the coach did not reflect or affirm it.", [("participant", "I really want to get my sugar down, I wrote it on the calendar.", "change"), ("coach", "Do you have a glucometer?", "closed question")]),
        ("Three closed questions in a row.", [("coach", "Did you walk?", "closed question"), ("participant", "No.", "neutral"), ("coach", "Was it the heat?", "closed question"), ("participant", "Yes.", "neutral"), ("coach", "Can you walk in the morning?", "closed question")]),
        ("Participant raised a medical concern. Help the coach respond with care and connect them to the right person today.", [("participant", "Oh, before you go, I've had a fever since last night. 101.", "neutral")]),
    ]
    print()
    for trigger, recent in moments:
        print("##", trigger)
        for model in MODELS:
            out, secs = post("/api/cue", {"model": model, "trigger": trigger, "recent": [{"speaker": s, "text": t, "tag": g} for s, t, g in recent]})
            print(f"  {model.split('/')[-1]:15s} {secs:4.1f}s  {out.get('cue')}  |  {out.get('say')}  {out.get('error', '')}")


if __name__ == "__main__":
    main()
