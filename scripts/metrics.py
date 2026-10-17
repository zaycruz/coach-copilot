"""Every headline number in README.md, rebuilt offline from the committed model outputs (runs/predictions/*.jsonl,
ids and model answers only) joined with the sponsor CSVs (labels). No network, no keys; seeded, so it is exact.

Usage: metrics.py <data_dir> [out=runs/metrics.json]
<data_dir> holds synthetic_data/*.csv and coaching_sessions_annomi.csv (see REPRODUCE.md).
The model outputs themselves come from the live engine (Tier B in REPRODUCE.md): evaluate.py, annomi_bench.py,
review_eval.py and flag_eval.py, exported text-free by export_predictions.py.
"""
import hashlib, json, math, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRED = os.path.join(ROOT, "runs", "predictions")
RED = {"medical", "distress", "social"}
COARSE = {"open question": "question", "closed question": "question", "ask permission": "question",
          "simple reflection": "reflection", "complex reflection": "reflection",
          "give information": "giving information or advice", "persuade": "giving information or advice"}
SUBTYPE = {"open question": "open", "closed question": "closed", "simple reflection": "simple", "complex reflection": "complex"}


def preds(name):
    return pd.read_json(os.path.join(PRED, name), lines=True, dtype=False)


def r4(x):
    return round(float(x), 4)


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": int(k), "n": int(n), "rate": r4(p), "ci95": [r4(max(0, c - h)), r4(min(1, c + h))]}


def boot(x, y, fn, reps=2000):
    rng, x, y = np.random.default_rng(7), np.asarray(x, float), np.asarray(y, float)
    vals = [fn(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(reps))]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def call_boot(hit, call_id, reps=2000):
    """Percentile bootstrap over calls: lines within a call are not independent, so resample whole calls."""
    g = pd.DataFrame({"hit": hit.to_numpy(), "call": call_id.to_numpy()}).groupby("call").hit.agg(["sum", "count"]).to_numpy(float)
    rng = np.random.default_rng(7)
    vals = [(lambda s: s[:, 0].sum() / s[:, 1].sum())(g[rng.integers(0, len(g), len(g))]) for _ in range(reps)]
    return [r4(np.percentile(vals, 2.5)), r4(np.percentile(vals, 97.5))]


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


def load(data):
    lines = pd.read_csv(f"{data}/synthetic_data/synthetic_call_transcripts.csv")
    calls = pd.read_csv(f"{data}/synthetic_data/synthetic_calls.csv")
    lines["gold"] = lines.coach_move.where(lines.speaker == "coach", lines.participant_talk)
    return lines.merge(calls[["call_id", "language"]], on="call_id"), calls


def tags(lines, split):
    t = preds(f"tags-{split}.jsonl").merge(lines[["call_id", "line_number", "speaker", "gold", "language"]], on=["call_id", "line_number"])
    out = {"calls": int(t.call_id.nunique()), "lines": len(t), "noAnswer": int(t.pred.isna().sum())}
    for sp, q in (("coach", "move"), ("participant", "talk")):
        g = t[t.speaker == sp]
        a = g[g.pred.notna()]
        hit = int((a.pred == a.gold).sum())
        out[q] = {"answered": len(a), "accuracy": r4(hit / len(a)), "accuracyCi95": call_boot(a.pred == a.gold, a.call_id),
                  "accuracyCountingNoAnswerWrong": r4((g.pred == g.gold).mean()),
                  # Baseline: always answer the most common human label.
                  "majorityBaseline": {"label": a.gold.mode()[0], "accuracy": r4((a.gold == a.gold.mode()[0]).mean())},
                  "byLanguage": {lang: r4((x.pred == x.gold).mean()) for lang, x in a.groupby("language")}}
    a = t[t.pred.notna()]
    hi = a.confidence >= 0.3
    out["calibration"] = {"shareAtLeast0.3": r4(hi.mean()), "accuracyAtLeast0.3": r4((a[hi].pred == a[hi].gold).mean()),
                          "accuracyBelow0.3": r4((a[~hi].pred == a[~hi].gold).mean())}
    return out, t


def flags(lines, split):
    f = preds(f"flags-{split}.jsonl").merge(lines[["call_id", "line_number", "text", "needs_human"]], on=["call_id", "line_number"])
    fired, urgent = f.flag.fillna("none") != "none", f.needs_human == "yes"
    fa = f[fired & ~urgent]
    return {"caught": wilson((fired & urgent).sum(), urgent.sum()),
            "caughtKinds": {k: int(v) for k, v in f[urgent].flag.value_counts().items()},
            "falseAlarmLines": wilson(len(fa), (~urgent).sum()), "precision": wilson((fired & urgent).sum(), fired.sum()),
            "urgentPrevalence": wilson(urgent.sum(), len(f)), "falseAlarmTexts": int(fa.text.nunique()),
            "falseAlarmCalls": wilson(fa.call_id.nunique(), f.call_id.nunique()),
            "falseAlarmDropout": int((fa.flag == "dropout").sum()), "falseAlarmRed": int(fa.flag.isin(RED).sum())}


def conduct(data):
    s = pd.read_csv(f"{data}/synthetic_data/synthetic_supervisor_ratings.csv")
    handled = s[s.safety_issue_handled.notna()].groupby("call_id").safety_issue_handled.agg(lambda v: "no" if (v == "no").any() else "yes")
    c = preds("conduct-supervised.jsonl").set_index("call_id").join(handled, how="inner")
    return wilson(((c.predicted_missed) == (c.safety_issue_handled == "no")).sum(), len(c))


def agreement(data, tag_rows):
    sup = pd.read_csv(f"{data}/synthetic_data/synthetic_supervisor_ratings.csv")
    coach = tag_rows[tag_rows.speaker == "coach"].sort_values(["call_id", "line_number"])
    app = coach.groupby("call_id").pred.apply(lambda s: score([m for m in s if isinstance(m, str)]))
    human = coach.groupby("call_id").gold.apply(lambda s: score(list(s)))
    wide = sup.pivot(index="call_id", columns="rater", values="overall_1_to_5").rename(columns={"Supervisor A": "a", "Supervisor B": "b"})
    t = wide.join(app.rename("app"), how="inner").join(human.rename("labels")).dropna(subset=["app"])
    t["mean"] = (t.a + t.b) / 2
    pairs = [pair("app score vs supervisor A", t.app, t.a), pair("app score vs supervisor B", t.app, t.b),
             pair("app score vs mean of A and B", t.app, t["mean"]), pair("supervisor A vs supervisor B", t.a, t.b),
             pair("human-label score vs mean of A and B", t.labels, t["mean"]), pair("app score vs human-label score", t.app, t.labels)]
    icc = {"pair": "supervisor A vs supervisor B (overall 1-5)", "icc21": round(icc21(t.a, t.b), 3),
           "ci95": boot(t.a, t.b, icc21), "mae": round(float((t.a - t.b).abs().mean()), 3)}
    cal = np.polyfit(t.app, t["mean"], 1)
    fit = np.clip(np.polyval(cal, t.app), 1, 5)
    mae = {"note": "app score mapped to 1-5 by least squares on these calls (slope, intercept below), then compared with each supervisor",
           "slope": round(float(cal[0]), 4), "intercept": round(float(cal[1]), 3),
           "maeVsA": round(float(np.abs(fit - t.a).mean()), 3), "maeVsB": round(float(np.abs(fit - t.b).mean()), 3), "maeAvsB": icc["mae"]}
    per_call = [{"callId": cid, "appScore": int(r.app), "labelScore": None if pd.isna(r.labels) else int(r.labels),
                 "supervisorA": int(r.a), "supervisorB": int(r.b)} for cid, r in t.sort_index().iterrows()]
    return {"calls": len(t), "correlations": pairs, "supervisorIcc": icc, "mae": mae}, per_call


def annomi(data):
    a = pd.read_csv(f"{data}/coaching_sessions_annomi.csv")
    r = preds("annomi-10expert.jsonl").merge(a, on=["session_id", "line_number"])
    coach = r.speaker == "counsellor"
    r["gold"] = r.counsellor_move.where(coach, r.client_talk)
    r["pred"] = r.clef.map(lambda p: COARSE.get(p, "other")).where(coach, r.clef)
    r["hit"] = r.pred == r.gold
    out = {"lines": len(r), "accuracy": r4(r.hit.mean()), "accuracyCi95": call_boot(r.hit, r.session_id), "expertAgreementPct": round(float(r.expert_agreement_pct.mean()), 1)}
    for sp, g in r.groupby("speaker"):
        bands = pd.cut(g.expert_agreement_pct, [0, 70, 90, 100], labels=["<=70", "80-90", "100"])
        out[sp] = {"lines": len(g), "accuracy": r4(g.hit.mean()), "expertAgreementPct": round(float(g.expert_agreement_pct.mean()), 1),
                   "byAgreement": {str(b): {"lines": len(x), "accuracy": r4(x.hit.mean())} for b, x in g.groupby(bands, observed=True)}}
    out["clientChangeOrSustainReadAsNeutral"] = int((~coach & r.gold.isin(["change", "sustain"]) & (r.pred == "neutral")).sum())
    sub = r[coach & r.hit]
    sub = sub.assign(sg=np.where(sub.gold == "question", sub.question_type, np.where(sub.gold == "reflection", sub.reflection_type, None)),
                     sp=sub.clef.map(SUBTYPE))
    for kind in ("question", "reflection"):
        # Only lines where Clef's own label has a subtype ("ask permission" counts as a question but has none).
        g = sub[(sub.gold == kind) & sub.sg.notna() & sub.sp.notna()]
        out[f"{kind}Subtype"] = {"lines": len(g), "accuracy": r4((g.sg == g.sp).mean())}
    return out


def review(tag_rows):
    v = preds("review-validation.jsonl").merge(tag_rows, on=["call_id", "line_number"])
    final = v.review.where(v.review.notna(), v.pred)
    ans = v[v.review.notna()]
    return {"lines": len(v), "answered": len(ans), "clefAlone": r4((v.pred == v.gold).mean()), "clefPlusReview": r4((final == v.gold).mean()),
            "fixed": int(((v.pred != v.gold) & (final == v.gold)).sum()), "broke": int(((v.pred == v.gold) & (final != v.gold)).sum()),
            "answeredClef": r4((ans.pred == ans.gold).mean()), "answeredReview": r4((ans.review == ans.gold).mean())}


def main():
    data = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "runs", "metrics.json")
    lines, _ = load(data)
    val, val_rows = tags(lines, "validation")
    dev, _ = tags(lines, "dev")
    agree, per_call = agreement(data, val_rows)
    m = {"tags": {"validation": val, "dev": dev},
         "flags": {"validation": flags(lines, "validation"), "dev": flags(lines, "dev"), "coachConduct": conduct(data)},
         "agreement": agree, "annomi": annomi(data), "review": review(val_rows)}
    m["agreementPerCallSha256"] = hashlib.sha256(json.dumps(per_call).encode()).hexdigest()
    json.dump(m, open(out_path, "w"), indent=1, ensure_ascii=False)
    json.dump({"perCall": per_call}, open(os.path.join(os.path.dirname(out_path), "agreement-per-call.json"), "w"), indent=1)
    print(json.dumps(m, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
