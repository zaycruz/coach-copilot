"""Turn the full local run outputs into the text-free prediction files committed under runs/predictions/.
They hold ids and model output only (no transcript text, no labels), so the repo carries no sponsor data,
and scripts/metrics.py rebuilds every headline number from them plus the sponsor CSVs.

Usage: export_predictions.py   (reads runs/*.jsonl and runs/flag-eval-v2.log, written by the model-query scripts)
"""
import json, os, random, re

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
OUT = os.path.join(RUNS, "predictions")
rows = lambda name: [json.loads(l) for l in open(os.path.join(RUNS, name))]


def write(name, recs):
    with open(os.path.join(OUT, name), "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(recs)} rows")


os.makedirs(OUT, exist_ok=True)
tags = rows("validation-clef-ctx2.jsonl")
for split, recs in (("validation", tags), ("dev", rows("dev-clef-ctx2.jsonl"))):
    write(f"tags-{split}.jsonl", [{"call_id": r["call_id"], "line_number": r["line_number"], "pred": r["pred"],
                                   "confidence": r["confidence"]} for r in recs])
for split in ("validation", "dev"):
    write(f"flags-{split}.jsonl", [{"call_id": r["call_id"], "line_number": r["line_number"], "flag": r["flag"]}
                                   for r in rows(f"{split}-clef-ctx2-flags.jsonl")])
write("annomi-10expert.jsonl", [{"session_id": r["session_id"], "line_number": r["line_number"], "clef": r["clef"]}
                                for r in rows("annomi-10expert.jsonl")])

# review_eval.py drew its sample with random.seed(23) from the low-confidence tag rows; redraw it for the line ids.
low = [r for r in tags if r["confidence"] is not None and r["confidence"] < 0.3]
random.seed(23)
sample = random.sample(low, 600)
review = rows("review-validation.jsonl")
assert all(s["gold"] == r["gold"] and s["pred"] == r["clef"] for s, r in zip(sample, review)), "review sample does not line up"
write("review-validation.jsonl", [{"call_id": s["call_id"], "line_number": s["line_number"], "review": r["review"]}
                                  for s, r in zip(sample, review)])

# flag_eval.py prints one line per supervisor-rated call: "<call_id> supervisor handled=<yes|no> predicted missed=<bool>".
conduct = [{"call_id": m[1], "predicted_missed": m[2] == "True"}
           for m in re.finditer(r"^(CALL\d+) supervisor handled=\S+\s+predicted missed=(True|False)", open(os.path.join(RUNS, "flag-eval-v2.log")).read(), re.M)]
write("conduct-supervised.jsonl", conduct)
