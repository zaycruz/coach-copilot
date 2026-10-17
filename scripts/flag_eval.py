"""Flag checks against labels.
1) Coach conduct: the calls whose supervisors rated safety_issue_handled (yes/no). A call is predicted
   'not handled' when any coach line gets conduct=missed_concern.
2) Participant flags: the validation lines that raised false alarms before, plus every needs_human=yes line.
Usage: flag_eval.py <data_dir>
"""
import json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

def turn(line, prev):
    body = json.dumps({"line": line, "previous": prev}).encode()
    req = urllib.request.Request("https://coach.raava.dev/api/turn", body, {"content-type": "application/json", "user-agent": "coach-copilot/0.1"})
    for _ in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception:
            pass
    return {}

data = sys.argv[1]
l = pd.read_csv(f"{data}/synthetic_data/synthetic_call_transcripts.csv")
s = pd.read_csv(f"{data}/synthetic_data/synthetic_supervisor_ratings.csv")
rated = s[s.safety_issue_handled.notna()].groupby("call_id").safety_issue_handled.agg(lambda v: "no" if (v == "no").any() else "yes")

def call_flags(cid):
    """Line by line, like the browser: each line's previous lines carry their flags."""
    rows = l[l.call_id == cid].sort_values("line_number").to_dict("records")
    seen, flags = [], []
    for r in rows:
        o = turn({"speaker": r["speaker"], "text": r["text"]}, seen[-2:])
        seen.append({"speaker": r["speaker"], "text": r["text"], "flag": o.get("flag") or None})
        if o.get("flag") not in (None, "none"):
            flags.append((r["speaker"], r["text"], o["flag"]))
    return flags

print("== coach conduct vs supervisor safety_issue_handled ==")
hit = 0
for cid, handled in rated.items():
    flags = call_flags(cid)
    missed = any(f == "missed_concern" for _, _, f in flags)
    hit += missed == (handled == "no")
    print(f"{cid} supervisor handled={handled:3s} predicted missed={missed!s:5s}", [(sp[0], f, t[:50]) for sp, t, f in flags])
print(f"agreement {hit}/{len(rated)}")

print("\n== participant flags on earlier false alarms and true cases ==")
v = pd.DataFrame([json.loads(x) for x in open("runs/validation-clef-ctx2.jsonl")])
v = v[v.speaker == "participant"].merge(l[["call_id", "line_number", "needs_human"]], on=["call_id", "line_number"])
fa = v[(v.safety.notna()) & (v.safety != "none") & (v.needs_human != "yes")].drop_duplicates("text")
tp = v[v.needs_human == "yes"]
def judge_row(r):
    rows = l[l.call_id == r.call_id].sort_values("line_number").to_dict("records")
    i = next(k for k, x in enumerate(rows) if x["line_number"] == r.line_number)
    return turn({"speaker": "participant", "text": r.text}, [{"speaker": p["speaker"], "text": p["text"]} for p in rows[max(0, i - 2):i]]).get("flag")
with ThreadPoolExecutor(8) as pool:
    fa_out = list(pool.map(judge_row, fa.itertuples()))
    tp_out = list(pool.map(judge_row, tp.itertuples()))
print(f"old false alarms (unique texts) still flagged: {sum(f not in (None, 'none') for f in fa_out)}/{len(fa)}")
for t, f in zip(fa.text, fa_out):
    if f not in (None, "none"): print("   still:", f, t[:80])
print(f"true cases caught: {sum(f not in (None, 'none') for f in tp_out)}/{len(tp)}", sorted(set(tp_out), key=str))
