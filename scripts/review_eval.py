"""Does the GLM review earn its place? A fixed random sample of validation lines where Clef's confidence < 0.3,
sent through the live /api/review exactly as the UI sends them. A review with no label keeps Clef's tag, as in the UI.

Usage: review_eval.py <data_dir> [n_lines]
Clef tags come from runs/validation-clef-ctx2.jsonl. Writes runs/review-validation.jsonl.
"""
import json, os, random, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm_bench import post

HERE = os.path.dirname(os.path.abspath(__file__))
data, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 600
lines = pd.read_csv(f"{data}/synthetic_data/synthetic_call_transcripts.csv")
by_call = {cid: g.sort_values("line_number").to_dict("records") for cid, g in lines.groupby("call_id")}
rows = [json.loads(l) for l in open(os.path.join(HERE, "..", "runs", "validation-clef-ctx2.jsonl"))]
low = [r for r in rows if r["confidence"] is not None and r["confidence"] < 0.3]
random.seed(23)
sample = random.sample(low, min(n, len(low)))
for r in sample:
    call = by_call[r["call_id"]]
    i = next(k for k, x in enumerate(call) if x["line_number"] == r["line_number"])
    r["previous"] = [{"speaker": x["speaker"], "text": x["text"]} for x in call[max(0, i - 2):i]]
print(f"validation: {len(low)}/{len(rows)} lines below 0.3 ({len(low) / len(rows):.0%}); sample {len(sample)}")


def review(r):
    out, secs = post("/api/review", {"line": {"speaker": r["speaker"], "text": r["text"]}, "previous": r["previous"],
                                     "probabilities": r["probabilities"]})
    return out.get("label"), secs


with ThreadPoolExecutor(8) as pool:
    res = list(pool.map(review, sample))

out = []
for r, (label, secs) in zip(sample, res):
    final = label or r["pred"]
    out.append({"speaker": r["speaker"], "gold": r["gold"], "clef": r["pred"], "review": label, "final": final, "secs": secs})
d = pd.DataFrame(out)
d.to_json(os.path.join(HERE, "..", "runs", "review-validation.jsonl"), orient="records", lines=True, force_ascii=False)
ans = d[d.review.notna()]
lat = sorted(ans.secs) or [0]
print(f"answered {len(ans)}/{len(d)}; review p50 {lat[len(lat) // 2]:.1f}s p95 {lat[int(len(lat) * .95)]:.1f}s")
for sp, g in [("all", d)] + list(d.groupby("speaker")):
    fixed = ((g.clef != g.gold) & (g.final == g.gold)).sum()
    broke = ((g.clef == g.gold) & (g.final != g.gold)).sum()
    print(f"{sp}: Clef alone {(g.clef == g.gold).mean():.1%}, Clef + review {(g.final == g.gold).mean():.1%} "
          f"of {len(g)}; review fixed {fixed}, broke {broke}")
print("on answered lines: Clef", f"{(ans.clef == ans.gold).mean():.1%}", "review", f"{(ans.review == ans.gold).mean():.1%}")
print("changes:", dict(Counter(zip(d[d.final != d.clef].clef, d[d.final != d.clef].final)).most_common(8)))
