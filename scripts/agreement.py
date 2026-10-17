"""How well does the app's coach score track the two supervisors? And flag recall / false alarms with Wilson CIs.

Calls: the supervisor-rated calls in the validation split (held out; never used for tuning). The app score is the
MITI-style score (scripts/score.py) over Clef's tags (runs/predictions/tags-validation.jsonl), no GLM review.
For reference, the same score over the human line labels shows what the score formula itself can reach.
Correlations get a percentile bootstrap 95% CI over calls (2,000 resamples, seed 7); A vs B also gets ICC(2,1).
Like-for-like: each supervisor is compared with the app and with the other supervisor (paired bootstrap of the
difference in r), and the app score, mapped to 1-5 with cross-fitting, gets ICC(2,1) against each supervisor.

Usage: agreement.py <data_dir>   -> writes src/agreement.json (served at GET /api/c5/agreement) and prints a summary.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import ROOT, agreement, annomi, conduct, cues, flags, load, stress, tags

data = sys.argv[1]
lines, _ = load(data)
val, rows = tags(lines, "validation")
agree, per_call = agreement(data, rows)
f = flags(lines, "validation")
out = {"about": "Coach score vs two supervisors on held-out (validation) rated calls. Synthetic data, Crane Lab.",
       "method": __doc__.split("Usage:")[0].strip(), **agree,
       "flags": {"recall": f["caught"], "falseAlarmsPerLine": f["falseAlarmLines"], "callsWithFalseAlarm": f["falseAlarmCalls"],
                 "coachConductAgreement": conduct(data), "byLanguage": f["byLanguage"],
                 "source": "validation split participant lines; coach conduct = calls supervisors rated for safety_issue_handled"},
       "tagAgreement": {q: {"accuracy": val[q]["accuracy"], "kappa": val[q]["labels"]["kappa"], "kappaByLanguage": val[q]["kappaByLanguage"],
                            "accuracyByLanguage": val[q]["byLanguage"]} for q in ("move", "talk")}
                       | {"annomiKappa": {sp: v["kappa"] for sp, v in annomi(data)["labels"].items()}},
       "safetyStress": {k: {"urgentCaught": m["urgent"]["caught"], "byLanguage": m["urgent"]["byLanguage"], "byKind": m["urgent"]["byKind"],
                            "hardNegativesFlagged": m["hardNegatives"]["falseAlarms"]}
                        for k, m in (("v4", stress("stress-set-v4.json", "stress-v4-flags.jsonl")), ("v4Before", stress("stress-set-v4.json", "stress-v4-flags-before.jsonl")),
                                     ("v3", stress("stress-set-v3.json", "stress-v3-flags.jsonl")), ("v2", stress("stress-set-v2.json", "stress-v2-flags.jsonl")))}
                       | {"note": "Independent hard urgent lines (EN/ES/Haitian Creole), each set written by a separate agent and committed before its run. "
                                  "v4 is held out for the current distress definition; v4Before is the previous one (Worker ec884907); v2 and v3 ran with the previous one. "
                                  "Haitian Creole is tested on the safety flag only."},
       "cues": cues(data, rows),
       "perCall": per_call}
json.dump(out, open(os.path.join(ROOT, "src", "agreement.json"), "w"), ensure_ascii=False, indent=1)
print(f"{agree['calls']} validation calls rated by both supervisors")
for p in agree["correlations"]:
    print(f"  {p['pair']:40s} r={p['pearson']:.3f} {p['pearsonCi95']}  rho={p['spearman']:.3f} {p['spearmanCi95']}")
print("  ICC(2,1) A vs B", agree["supervisorIcc"], "\n  MAE", agree["mae"])
for k, v in out["flags"].items():
    print(f"  {k}: {v}")
