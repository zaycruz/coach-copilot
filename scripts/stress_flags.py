"""Run the hand-written red-flag stress set (scripts/stress-set.json) through the live /api/turn, one participant line
at a time with no previous lines, as the flag request sees it. No key. 60 turns (120 Clef calls), 2 at a time, waits out 429.

Usage: stress_flags.py [baseUrl]   (default $COACH_COPILOT_URL or https://coach.raava.dev)
Writes runs/predictions/stress-flags.jsonl (id, flag, tag); scripts/metrics.py scores it. Do not run during a demo or panel run.
"""
import json, os, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("COACH_COPILOT_URL", "https://coach.raava.dev")).rstrip("/")
lines = json.load(open(os.path.join(ROOT, "scripts", "stress-set.json")))["lines"]


def turn(line):
    body = json.dumps({"line": {"speaker": "participant", "text": line["text"]}, "previous": []}).encode()
    req = urllib.request.Request(BASE + "/api/turn", body, {"content-type": "application/json", "user-agent": "coach-copilot-stress/1.0"})
    for _ in range(8):
        try:
            o = json.load(urllib.request.urlopen(req, timeout=60))
            return {"id": line["id"], "flag": o.get("flag"), "tag": o.get("tag")}
        except urllib.error.HTTPError as err:
            if err.code not in (429, 502):
                raise
            time.sleep(int(err.headers.get("retry-after") or 10))
    return {"id": line["id"], "flag": None, "tag": None}


with ThreadPoolExecutor(2) as pool:
    out = list(pool.map(turn, lines))
with open(os.path.join(ROOT, "runs", "predictions", "stress-flags.jsonl"), "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{sum(r['flag'] is not None for r in out)}/{len(out)} answered; wrote runs/predictions/stress-flags.jsonl")
