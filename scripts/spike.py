"""Spike: tag every coach line of a few labelled synthetic calls with one Clef request per call."""
import json, os, sys, time, urllib.request
import pandas as pd

DATA = sys.argv[1]
URL = "https://coach-copilot-clef.zay-259.workers.dev/judge"
KEY = open(os.path.join(os.path.dirname(__file__), "..", ".env")).read().split("=", 1)[1].strip()

MOVES = {
    "open question": "Asks a question that invites a long, free answer (what/how/tell me).",
    "closed question": "Asks a yes/no or short-fact question.",
    "simple reflection": "Repeats or lightly rephrases what the participant just said, adding no new meaning.",
    "complex reflection": "Reflects back with added meaning: feelings, ambivalence, a guess at what is underneath.",
    "affirm": "Praises the participant's effort, strengths or values.",
    "seek collaboration": "Invites the participant to plan or decide together.",
    "emphasize autonomy": "Says the choice is the participant's.",
    "ask permission": "Asks before giving advice or information or changing topic.",
    "give information": "Gives facts or explains, without pushing the participant to act.",
    "persuade": "Gives advice or argues for change the participant did not ask for.",
    "confront": "Disagrees, criticises, shames, warns or argues with the participant.",
    "other": "Greetings, small talk, logistics, fillers, scheduling, anything else.",
}

def post(payload):
    req = urllib.request.Request(URL, json.dumps(payload).encode(), {"content-type": "application/json", "authorization": f"Bearer {KEY}", "user-agent": "coach-copilot/0.1"})
    return json.load(urllib.request.urlopen(req, timeout=120))

lines = pd.read_csv(f"{DATA}/synthetic_data/synthetic_call_transcripts.csv")
calls = pd.read_csv(f"{DATA}/synthetic_data/synthetic_calls.csv")
model = sys.argv[2] if len(sys.argv) > 2 else "@cf/cloudflare/clef"
ids = calls[calls.labelled == "yes"].call_id.iloc[:int(sys.argv[3]) if len(sys.argv) > 3 else 3]
hit = tot = 0
for cid in ids:
    call = lines[lines.call_id == cid]
    state = {"transcript": [f"L{r.line_number} {r.speaker}: {r.text}" for r in call.itertuples()]}
    coach = call[(call.speaker == "coach") & call.coach_move.notna()].iloc[:64]
    questions = {f"L{r.line_number}": {"type": "choice", "instructions": f"Motivational-interviewing code for coach line L{r.line_number}, judged in context of the line before it.", "criteria": MOVES} for r in coach.itertuples()}
    t = time.time(); out = post({"model": model, "state": state, "questions": questions})
    if "result" not in out: print(cid, out); continue
    ans = out["result"].get("answers", {})
    if False: print(json.dumps(out["result"], ensure_ascii=False)[:800])
    for r in coach.itertuples():
        a = ans.get(f"L{r.line_number}", {}); pred = a.get("choice") if isinstance(a, dict) else a
        tot += 1; hit += pred == r.coach_move
        if pred != r.coach_move: print(f"  miss {cid} L{r.line_number}: gold={r.coach_move} pred={pred} | {r.text[:70]}")
    print(cid, f"{len(coach)} lines", f"{time.time()-t:.1f}s", f"worker {out['ms']}ms")
print(f"{model}: {hit}/{tot} = {hit/max(tot,1):.1%}")
