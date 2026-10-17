"""Pre-play one safety call through the live engine, as the demo page's Replay does, and store the result for
GET /api/c5/demo: every line's Clef tag and flag, the cues GLM wrote at the trigger moments, the banners and the score.
The call is CALL05423 (validation split, chest pain), so nothing in it was used for tuning.

Usage: preplay.py [call_id]   -> writes src/demo-safety-call.json. Makes one /api/turn per line and one /api/cue per cue.
"""
import json, os, sys, time, urllib.error, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import score

BASE, HERE = os.environ.get("COACH_COPILOT_URL", "https://coach.raava.dev").rstrip("/"), os.path.dirname(os.path.abspath(__file__))
CALL = sys.argv[1] if len(sys.argv) > 1 else "CALL05423"
FAMILY = {"open question": "q", "closed question": "q", "simple reflection": "r", "complex reflection": "r"}
BANNER = {"medical": ("red", "Medical concern: tell the care team today"),
          "distress": ("red", "Emotional distress: check in and connect to support today"),
          "social": ("red", "Unmet basic need: connect to a navigator today"),
          "dropout": ("amber", "Dropout risk: participant may leave the program"),
          "missed_concern": ("amber", "Coach did not route a concern the participant raised"),
          "medical_advice": ("amber", "Coach gave medical advice outside their role")}
CUE_GAP = 3


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"content-type": "application/json", "user-agent": "coach-copilot/0.1"})
    for _ in range(6):
        try:
            return json.load(urllib.request.urlopen(req, timeout=90))
        except urllib.error.HTTPError as err:
            if err.code != 429:
                raise
            time.sleep(int(err.headers.get("retry-after") or 20))
    raise RuntimeError("rate limited")


def trigger_for(lines, i, last_cue_at):
    """public/index.html checkTriggers, line for line."""
    l = lines[i]
    if l["flag"] == "missed_concern":
        return "The participant raised a concern and the coach moved on without routing it. Help the coach go back to it now."
    if l["flag"] == "medical_advice":
        return "The coach just gave medical advice outside their role. Help the coach walk it back and refer to the care team."
    if l["flag"] == "dropout":
        return "The participant is talking about leaving the program. Help the coach explore it without pressure."
    if l["flag"] not in (None, "none"):
        return f"Participant raised a {l['flag']} concern. Help the coach respond with care and connect them to the right person today."
    if l["speaker"] != "coach":
        return None
    coach = [x for x in lines[:i + 1] if x["speaker"] == "coach" and x["tag"]]
    last, before, t = [x["tag"] for x in coach[-4:]], lines[i - 1] if i else None, None
    if l["tag"] == "confront":
        t = "The coach just confronted the participant."
    elif l["tag"] == "persuade" and not any(x["tag"] == "ask permission" for x in coach[-3:-1]):
        t = "The coach just told the participant what to do without asking permission."
    elif before and before["speaker"] == "participant" and before["tag"] == "change" and FAMILY.get(l["tag"]) != "r" and l["tag"] != "affirm":
        t = "The participant voiced change talk and the coach did not reflect or affirm it."
    elif len(last) == 4 and all(FAMILY.get(x) == "q" for x in last):
        t = "The coach has asked four questions in a row without a reflection."
    elif len(last) >= 3 and all(x == "closed question" for x in last[-3:]):
        t = "Three closed questions in a row."
    if t and sum(x["speaker"] == "coach" for x in lines[:i + 1]) - last_cue_at < CUE_GAP:
        return None
    return t


src = next(c for c in json.load(open(os.path.join(HERE, "..", "public", "demo-calls.json")))["calls"] if c["call_id"] == CALL)
lines, cues, last_cue_at = [], [], -99
for r in src["lines"]:
    prev = [{"speaker": x["speaker"], "text": x["text"], "flag": x["flag"]} for x in lines[-2:]]
    o = post("/api/turn", {"line": {"speaker": r["speaker"], "text": r["text"]}, "previous": prev})
    lines.append({"n": r["line_number"], "t": r["time_in_call"], "speaker": r["speaker"], "text": r["text"],
                  "tag": o.get("tag"), "confidence": o.get("confidence"), "flag": o.get("flag") or "none",
                  "supervisor": {"move": r["coach_move"]} if r["speaker"] == "coach" else {"talk": r["participant_talk"], "needsHuman": r["needs_human"] == "yes"}})
    i = len(lines) - 1
    trig = trigger_for(lines, i, last_cue_at)
    if trig:
        last_cue_at = sum(x["speaker"] == "coach" for x in lines)
        recent = [{"speaker": x["speaker"], "text": x["text"], "tag": x["tag"]} for x in lines[max(0, i - 7):i + 1]]
        c = post("/api/cue", {"trigger": trig, "recent": recent})
        if c.get("cue"):
            cues.append({"afterLine": r["line_number"], "t": r["time_in_call"], "trigger": trig, "cue": c["cue"], "say": c.get("say") or ""})

moves = [x["tag"] for x in lines if x["speaker"] == "coach" and x["tag"]]
banners = [{"line": x["n"], "t": x["t"], "flag": x["flag"], "level": BANNER[x["flag"]][0], "text": BANNER[x["flag"]][1], "quote": x["text"]}
           for x in lines if x["flag"] in BANNER]
out = {"about": "A safety call pre-played through the live engine (Clef tags, GLM cues). Synthetic data, Crane Lab; nobody is real.",
       "callId": CALL, "survivorId": f"c5-{src['participant_id']}", "language": src["language"], "goalTopic": src["goal_topic"],
       "date": src["date"], "playedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "engine": {"tags": "@cf/cloudflare/clef", "cues": "@cf/zai-org/glm-5.3-flash"},
       "score": score(moves), "supervisorScore": score([x["supervisor"]["move"] for x in lines if x["speaker"] == "coach"]),
       "tagAgreement": round(sum(x["tag"] == (x["supervisor"].get("move") or x["supervisor"].get("talk")) for x in lines) / len(lines), 3),
       "banners": banners, "cues": cues, "lines": lines}
json.dump(out, open(os.path.join(HERE, "..", "src", "demo-safety-call.json"), "w"), ensure_ascii=False, indent=1)
print(CALL, "score", out["score"], "supervisor-label score", out["supervisorScore"], "tag agreement", out["tagAgreement"])
print("banners:", [(b["t"], b["flag"]) for b in banners]); print("cues:", [(c["t"], c["cue"]) for c in cues])
