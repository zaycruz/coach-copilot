"""Run the hand-written red-flag stress set (scripts/stress-set.json) through the live /api/turn, one participant line
at a time with no previous lines, as the flag request sees it. No key. 60 turns (120 Clef calls), 2 at a time, waits out 429.

Usage: stress_flags.py [baseUrl] [set] [out]   (defaults: $COACH_COPILOT_URL or https://coach.raava.dev, stress-set.json, stress-flags.jsonl)
v2: stress_flags.py https://coach.raava.dev stress-set-v2.json stress-v2-flags.jsonl
Writes runs/predictions/<out> (id, flag, tag, rule); scripts/metrics.py scores it. Do not run during a demo or panel run.
"""
import json, os, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("COACH_COPILOT_URL", "https://coach.raava.dev")).rstrip("/")
SET, OUT = (sys.argv[2] if len(sys.argv) > 2 else "stress-set.json"), (sys.argv[3] if len(sys.argv) > 3 else "stress-flags.jsonl")
lines = json.load(open(os.path.join(ROOT, "scripts", SET)))["lines"]


def turn(line):
    body = json.dumps({"line": {"speaker": "participant", "text": line["text"]}, "previous": []}).encode()
    req = urllib.request.Request(BASE + "/api/turn", body, {"content-type": "application/json", "user-agent": "coach-copilot-stress/1.0"})
    for _ in range(8):
        try:
            o = json.load(urllib.request.urlopen(req, timeout=60))
            return {"id": line["id"], "flag": o.get("flag"), "tag": o.get("tag"), **({"rule": o["rule"]} if o.get("rule") else {})}
        except urllib.error.HTTPError as err:
            if err.code not in (429, 502):
                raise
            time.sleep(int(err.headers.get("retry-after") or 10))
    return {"id": line["id"], "flag": None, "tag": None}


with ThreadPoolExecutor(2) as pool:
    out = list(pool.map(turn, lines))
with open(os.path.join(ROOT, "runs", "predictions", OUT), "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{sum(r['flag'] is not None for r in out)}/{len(out)} answered; wrote runs/predictions/{OUT}")
