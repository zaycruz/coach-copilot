"""Clef vs the AnnoMI 10-expert lines (428 lines, 7 sessions), where experts agree ~84% of the time.

The shipped payload runs unchanged (its setting still says cancer-survivor coaching). Context is the two
previous lines of the full session. AnnoMI codes counsellor lines 4-way, so the 12 Clef moves collapse:
  open/closed question, ask permission       -> question   (AnnoMI codes "Is it okay if I...?" as a question)
  simple/complex reflection                  -> reflection
  give information, persuade                 -> giving information or advice
  affirm, seek collaboration, emphasize autonomy, confront, other -> other
Subtypes (open/closed, simple/complex) are scored only where gold and prediction share the coarse label;
AnnoMI fills those columns on rows where they do not apply.
Usage: annomi_bench.py <data_dir>
"""
import json, os, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import post
from payload import turn_state, turn_questions

COARSE = {"open question": "question", "closed question": "question", "ask permission": "question",
          "simple reflection": "reflection", "complex reflection": "reflection",
          "give information": "giving information or advice", "persuade": "giving information or advice"}
SUBTYPE = {"open question": "open", "closed question": "closed", "simple reflection": "simple", "complex reflection": "complex"}
SPEAKER = {"counsellor": "coach", "client": "participant"}

a = pd.read_csv(f"{sys.argv[1]}/coaching_sessions_annomi.csv").sort_values(["session_id", "line_number"])
turns = []
for _, s in a.groupby("session_id"):
    rows = s.to_dict("records")
    for i, r in enumerate(rows):
        if r["expert_raters"] != 10:
            continue
        line = {"speaker": SPEAKER[r["speaker"]], "text": r["text"]}
        prev = [{"speaker": SPEAKER[p["speaker"]], "text": p["text"]} for p in rows[max(0, i - 2):i]]
        turns.append(r | {"line": line, "state": turn_state(line, prev)})


def judge(t):
    out = post({"state": t["state"], "questions": turn_questions(t["line"]["speaker"])})
    q = "move" if t["line"]["speaker"] == "coach" else "talk"
    return (((out.get("result") or {}).get("answers") or {}).get(q) or {}).get("choice")


with ThreadPoolExecutor(32) as pool:
    preds = list(pool.map(judge, turns))

rows = []
for t, p in zip(turns, preds):
    coach = t["line"]["speaker"] == "coach"
    gold = t["counsellor_move"] if coach else t["client_talk"]
    pred = COARSE.get(p, "other") if coach and p else p
    sub_gold = (t["question_type"] if gold == "question" else t["reflection_type"] if gold == "reflection" else None) if coach else None
    rows.append({"session_id": t["session_id"], "line_number": t["line_number"], "speaker": t["speaker"], "text": t["text"],
                 "gold": gold, "clef": p, "pred": pred, "hit": pred == gold, "agreement": t["expert_agreement_pct"],
                 "sub_gold": sub_gold, "sub_pred": SUBTYPE.get(p) if pred == gold and sub_gold else None})
r = pd.DataFrame(rows)
os.makedirs("runs", exist_ok=True)
r.to_json("runs/annomi-10expert.jsonl", orient="records", lines=True, force_ascii=False)

print(f"no answer: {r.clef.isna().sum()}/{len(r)}")
for sp, g in r.groupby("speaker"):
    print(f"{sp}: Clef {g.hit.mean():.1%} ({g.hit.sum()}/{len(g)}) vs expert agreement {g.agreement.mean():.1f}%")
    bands = pd.cut(g.agreement, [0, 70, 90, 100], labels=["<=70%", "80-90%", "100%"])
    for band, b in g.groupby(bands, observed=True):
        print(f"   experts {band}: {b.hit.mean():.1%} of {len(b)}")
    print("   misses:", dict(Counter(zip(g[~g.hit].gold, g[~g.hit].pred)).most_common(6)))
s = r[r.sub_pred.notna()]
for kind, g in s.groupby(s.sub_gold.map(lambda v: "question" if v in ("open", "closed") else "reflection")):
    print(f"{kind} subtype (where coarse label matched): {(g.sub_gold == g.sub_pred).mean():.1%} of {len(g)}")
print(f"all 428: {r.hit.mean():.1%}")
