"""Live E2E and drift check against a deployed Worker. No key, no GLM calls, about 12 Clef calls.

Checks: health; a red flag (EN chemo fever, ES chemo fever) and two everyday lines that must not flag; a coach line after a
red flag that must come back missed_concern; the shape of /api/c5/calls, survivors, demo; and that every number served by
/api/c5/agreement equals the repo's runs/metrics.json (the drift check). Also measures latency: the Worker's own `ms`
(Clef time inside the Worker) against the client round trip.

Usage: e2e_live.py [baseUrl]   (default $COACH_COPILOT_URL or https://coach.raava.dev)
Writes runs/e2e-live-<UTC stamp>.json and exits 1 if any check fails. Do not run it during a demo or a panel run.
"""
import json, os, statistics, sys, time, urllib.error, urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("COACH_COPILOT_URL", "https://coach.raava.dev")).rstrip("/")
checks, timings = [], []


def req(path, body=None):
    r = urllib.request.Request(BASE + path, json.dumps(body).encode() if body is not None else None,
                               {"content-type": "application/json", "user-agent": "coach-copilot-e2e/1.0"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(r, timeout=60) as res:
            out = res.status, json.load(res)
    except urllib.error.HTTPError as err:
        out = err.code, json.loads(err.read() or b"{}")
    return (*out, round((time.perf_counter() - t) * 1000))


def check(name, ok, got):
    checks.append({"check": name, "ok": bool(ok), "got": got})
    print(("PASS " if ok else "FAIL ") + name, "" if ok else json.dumps(got, ensure_ascii=False)[:300])


def turn(name, line, previous, want_flag):
    status, o, wall = req("/api/turn", {"line": line, "previous": previous})
    if status == 200:
        timings.append({"line": name, "speaker": line["speaker"], "workerMs": o["ms"], "roundTripMs": wall})
    check(f"turn: {name} -> flag {want_flag}", status == 200 and o.get("flag") == want_flag, {"status": status, **o})
    return o


status, o, _ = req("/api/health")
check("health", status == 200 and o.get("ok") is True, o)

prev = [{"speaker": "coach", "text": "How did the walking go this week?"},
        {"speaker": "participant", "text": "I walked three days, but my knees hurt after."}]
fever = {"speaker": "participant", "text": "Oh, before you go, I've had a fever since last night. I'm still on chemo."}
turn("EN fever on chemo", fever, prev, "medical")
turn("ES fever on chemo", {"speaker": "participant", "text": "Tengo fiebre desde anoche y sigo con la quimio."}, prev, "medical")
turn("everyday ache", {"speaker": "participant", "text": "My back is acting up again, but I still did my walks."}, prev, "none")
o = turn("ES passive ideation (red-flag rule)", {"speaker": "participant", "text": "Ya empecé a regalar mis cosas. No me veo aquí por mucho tiempo, y la verdad tampoco quiero."}, prev, "distress")
check("rule: ES passive ideation hit ideation-es", o.get("rule") == "ideation-es", o)
check("distress flag carries the protocol action", o.get("action") == "Use your suicide-risk screening protocol now.", o)
turn("dislikes the goal", {"speaker": "participant", "text": "Water just doesn't do it for me."}, prev, "none")
turn("coach moves on after the flag", {"speaker": "coach", "text": "Okay. So, about the walking goal."},
     [prev[0], {**fever, "flag": "medical"}], "missed_concern")
turn("coach line, no flag before", {"speaker": "coach", "text": "You kept going even with sore knees."}, prev, "none")
for i in range(4):  # a few more for a latency median
    turn(f"latency sample {i + 1}", {"speaker": "coach", "text": "What would make next week a little easier?"}, prev, "none")

status, bad, _ = req("/api/turn", {"line": {"speaker": "nurse", "text": "hi"}})
check("turn: bad speaker -> 400", status == 400, bad)

status, calls, _ = req("/api/c5/calls")
check("c5/calls: 27 calls with score, status, flags", status == 200 and len(calls) == 27
      and all({"callId", "survivorId", "score", "status", "flags"} <= set(c) for c in calls), {"status": status, "n": len(calls)})
status, people, _ = req("/api/c5/survivors")
check("c5/survivors: rows with survivorId and status", status == 200 and people and all({"survivorId", "status"} <= set(p) for p in people),
      {"status": status, "n": len(people)})
status, missing, _ = req("/api/c5/calls/CALL99999")
check("c5/calls/unknown -> 404", status == 404, missing)
status, demo, _ = req("/api/c5/demo")
check("c5/demo: pre-played safety call with a red banner and cues", status == 200 and demo.get("callId") == "CALL05423"
      and any(b["level"] == "red" for b in demo.get("banners", [])) and demo.get("cues"), {"status": status, "callId": demo.get("callId")})

# Drift check: what the live API serves vs what the repo recomputes offline (scripts/reproduce.sh).
status, live, _ = req("/api/c5/agreement")
repo = json.load(open(os.path.join(ROOT, "runs", "metrics.json")))
check("c5/agreement: 200", status == 200, {"status": status})
if status == 200:
    check("drift: correlations equal runs/metrics.json", live["correlations"] == repo["agreement"]["correlations"],
          {"live": [c["pearson"] for c in live["correlations"]], "repo": [c["pearson"] for c in repo["agreement"]["correlations"]]})
    check("drift: like-for-like, cross-fitted ICC and language split equal",
          all(live.get(k) == repo["agreement"][k] for k in ("likeForLike", "crossFittedIcc", "byLanguage")),
          {k: live.get(k) for k in ("likeForLike", "crossFittedIcc")})
    check("drift: supervisor ICC and MAE equal", live["supervisorIcc"] == repo["agreement"]["supervisorIcc"] and live["mae"] == repo["agreement"]["mae"],
          {"live": [live["supervisorIcc"], live["mae"]]})
    f = repo["flags"]
    check("drift: flag numbers equal", live["flags"]["recall"] == f["validation"]["caught"] and live["flags"]["falseAlarmsPerLine"] == f["validation"]["falseAlarmLines"]
          and live["flags"]["callsWithFalseAlarm"] == f["validation"]["falseAlarmCalls"] and live["flags"]["coachConductAgreement"] == f["coachConduct"],
          live["flags"])
    check("drift: cues and safety stress equal runs/metrics.json", live.get("cues") == repo["cues"]
          and live.get("safetyStress", {}).get("v4", {}).get("urgentCaught") == repo["stressV4"]["urgent"]["caught"]
          and live.get("tagAgreement", {}).get("move", {}).get("kappa") == repo["tags"]["validation"]["move"]["labels"]["kappa"],
          {"cues": (live.get("cues") or {}).get("precision"), "v4": live.get("safetyStress", {}).get("v4", {}).get("urgentCaught")})
    per_call = json.load(open(os.path.join(ROOT, "runs", "agreement-per-call.json")))["perCall"]
    check("drift: per-call table equal", live["perCall"] == per_call, {"liveRows": len(live["perCall"]), "repoRows": len(per_call)})

med = lambda k, rows: statistics.median(t[k] for t in rows) if rows else None
latency = {sp: {"lines": len(rows), "workerMsP50": med("workerMs", rows), "roundTripMsP50": med("roundTripMs", rows)}
           for sp in ("coach", "participant") for rows in [[t for t in timings if t["speaker"] == sp]]}
stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
out = {"ranAt": stamp, "base": BASE, "ranFrom": os.environ.get("E2E_RAN_FROM", "client"), "passed": sum(c["ok"] for c in checks),
       "total": len(checks), "latency": latency, "timings": timings, "checks": checks}
path = os.path.join(ROOT, "runs", f"e2e-live-{stamp}.json")
json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
print(f"{out['passed']}/{out['total']} passed; latency {json.dumps(latency)}; wrote {os.path.relpath(path, ROOT)}")
sys.exit(0 if out["passed"] == out["total"] else 1)
