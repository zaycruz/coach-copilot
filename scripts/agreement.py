"""How well does the app's coach score track the two supervisors? And flag recall / false alarms with Wilson CIs.

Calls: the supervisor-rated calls in the validation split (held out; never used for tuning). The app score is the
live screen's MITI-style arithmetic over Clef's tags (runs/validation-clef-ctx2.jsonl), no GLM review.
For reference, the same arithmetic over the human line labels shows what the score formula itself can reach.
Correlations get a percentile bootstrap 95% CI over calls (2,000 resamples, seed 7); A vs B also gets ICC(2,1).

Usage: agreement.py <data_dir>   -> writes src/agreement.json (served at GET /api/c5/agreement) and prints a summary.
"""
import json, math, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import score

HERE = os.path.dirname(os.path.abspath(__file__))
def wilson(k, n, z=1.96):
    if not n:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": int(k), "n": int(n), "rate": round(p, 4), "ci95": [round(max(0, c - h), 4), round(min(1, c + h), 4)]}


def boot(x, y, fn, reps=2000):
    rng, x, y = np.random.default_rng(7), np.asarray(x, float), np.asarray(y, float)
    vals = [fn(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(reps))]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


pearson = lambda x, y: float(np.corrcoef(x, y)[0, 1])
spearman = lambda x, y: pearson(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def icc21(a, b):
    """Two-way random, absolute agreement, single rater (Shrout & Fleiss ICC(2,1))."""
    m = np.column_stack([a, b]).astype(float)
    n, k = m.shape
    g, rows, cols = m.mean(), m.mean(1), m.mean(0)
    msr = k * ((rows - g) ** 2).sum() / (n - 1)
    msc = n * ((cols - g) ** 2).sum() / (k - 1)
    mse = ((m - rows[:, None] - cols[None, :] + g) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def pair(name, x, y):
    return {"pair": name, "n": len(x), "pearson": round(pearson(x, y), 3), "pearsonCi95": boot(x, y, pearson),
            "spearman": round(spearman(x, y), 3), "spearmanCi95": boot(x, y, spearman)}


data = sys.argv[1]
lines = pd.read_csv(f"{data}/synthetic_data/synthetic_call_transcripts.csv")
sup = pd.read_csv(f"{data}/synthetic_data/synthetic_supervisor_ratings.csv")
pred = pd.DataFrame([json.loads(l) for l in open(os.path.join(HERE, "..", "runs", "validation-clef-ctx2.jsonl"))])
coach = pred[pred.speaker == "coach"].sort_values(["call_id", "line_number"])
app = coach.groupby("call_id").pred.apply(lambda s: score([m for m in s if isinstance(m, str)]))
human = coach.groupby("call_id").gold.apply(lambda s: score(list(s)))
wide = sup.pivot(index="call_id", columns="rater", values="overall_1_to_5").rename(columns={"Supervisor A": "a", "Supervisor B": "b"})
t = wide.join(app.rename("app"), how="inner").join(human.rename("labels")).dropna(subset=["app"])
t["mean"] = (t.a + t.b) / 2

pairs = [pair("app score vs supervisor A", t.app, t.a), pair("app score vs supervisor B", t.app, t.b),
         pair("app score vs mean of A and B", t.app, t["mean"]), pair("supervisor A vs supervisor B", t.a, t.b),
         pair("human-label score vs mean of A and B", t.labels, t["mean"]), pair("app score vs human-label score", t.app, t.labels)]
icc = {"pair": "supervisor A vs supervisor B (overall 1-5)", "icc21": round(icc21(t.a, t.b), 3),
       "ci95": boot(t.a, t.b, lambda x, y: icc21(x, y)), "mae": round(float((t.a - t.b).abs().mean()), 3)}
cal = np.polyfit(t.app, t["mean"], 1)
fit = np.clip(np.polyval(cal, t.app), 1, 5)
mae = {"note": "app score mapped to 1-5 by least squares on these calls (slope, intercept below), then compared with each supervisor",
       "slope": round(float(cal[0]), 4), "intercept": round(float(cal[1]), 3),
       "maeVsA": round(float(np.abs(fit - t.a).mean()), 3), "maeVsB": round(float(np.abs(fit - t.b).mean()), 3),
       "maeAvsB": icc["mae"]}

# Flags, from the evaluate.py --flags-only run and flag_eval.py (supervisor safety_issue_handled).
fl = pd.DataFrame([json.loads(l) for l in open(os.path.join(HERE, "..", "runs", "validation-clef-ctx2-flags.jsonl"))])
fired = fl.flag.fillna("none") != "none"
urgent = fl.gold_safety == "yes"
flags = {"recall": wilson((fired & urgent).sum(), urgent.sum()),
         "falseAlarmsPerLine": wilson((fired & ~urgent).sum(), (~urgent).sum()),
         "callsWithFalseAlarm": wilson(fl[fired & ~urgent].call_id.nunique(), fl.call_id.nunique()),
         "coachConductAgreement": wilson(8, 8),
         "source": "validation split participant lines; coach conduct = 8 calls supervisors rated for safety_issue_handled (runs/flag-eval-v2.log)"}

per_call = [{"callId": cid, "appScore": int(r.app), "labelScore": None if pd.isna(r.labels) else int(r.labels),
             "supervisorA": int(r.a), "supervisorB": int(r.b)} for cid, r in t.sort_index().iterrows()]
out = {"about": "Coach score vs two supervisors on held-out (validation) rated calls. Synthetic data, Crane Lab.",
       "method": __doc__.split("Usage:")[0].strip(), "calls": len(t), "correlations": pairs, "supervisorIcc": icc,
       "mae": mae, "flags": flags, "perCall": per_call}
json.dump(out, open(os.path.join(HERE, "..", "src", "agreement.json"), "w"), ensure_ascii=False, indent=1)
print(f"{len(t)} validation calls rated by both supervisors")
for p in pairs:
    print(f"  {p['pair']:40s} r={p['pearson']:.3f} {p['pearsonCi95']}  rho={p['spearman']:.3f} {p['spearmanCi95']}")
print("  ICC(2,1) A vs B", icc, "\n  MAE", mae)
for k, v in flags.items():
    print(f"  {k}: {v}")
