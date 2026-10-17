"""Throttled flag re-validation: judge the participant flag for one split's lines, or for stress sets, under the current
FLAG definition (scripts/payload.py), through the Worker's keyed /judge route. It fills the same cache that
evaluate.py --flags-only reads (runs/cache-flag-clef-<version>.jsonl), so evaluate.py then scores the split with 0 new calls.

Throttle, so the shared Workers AI quota keeps headroom for the live app:
- at most THREADS (default 3) requests in flight and MIN_GAP seconds (default 0.25) between request starts;
- a rate-limited answer (HTTP 429, or Workers AI error 3021 in the body) pauses everything for 60 s;
- if more than 2 of the last 50 answers were rate-limited after a pause, or 3 pauses happen, it stops (exit 3): the
  quota margin is gone, so the run is left for later. The cache keeps what was done, and a re-run resumes;
- while a judges-panel evidence window is open (judges-panel/runs/.state/window.json), it waits.

Usage: flag_revalidate.py split <data_dir> <dev|validation>
       flag_revalidate.py stress <set.json> [<set.json> ...]   -> runs/stress-<set>-flags-<version>.jsonl (id, flag, rule, modelFlag)
"""
import collections, json, os, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import URL, api_key, split_of, HERE
from payload import FLAG, FLAG_INSTRUCTIONS, flag_request
from redflags import red_rule
import hashlib

THREADS, MIN_GAP = int(os.environ.get("THREADS", 3)), float(os.environ.get("MIN_GAP", 0.25))
VERSION = hashlib.sha1(json.dumps([FLAG, FLAG_INSTRUCTIONS], sort_keys=True).encode()).hexdigest()[:8]
CACHE = os.path.join(HERE, "..", "runs", f"cache-flag-clef-{VERSION}.jsonl")
WINDOW = os.path.join(HERE, "..", "..", "judges-panel", "runs", ".state", "window.json")
lock, recent, state = threading.Lock(), collections.deque(maxlen=50), {"next": 0.0, "pauses": 0, "pause_until": 0.0, "stop": False}


def window_open() -> bool:
    try:
        w = json.load(open(WINDOW))
        os.kill(int(w["pid"]), 0)
        return True
    except (OSError, ValueError, KeyError):
        return False


def gate() -> None:
    while True:
        while window_open():
            print("  evidence window open, waiting", flush=True)
            time.sleep(60)
        with lock:
            if state["stop"]:
                raise SystemExit(3)
            now = time.time()
            start = max(now, state["next"], state["pause_until"])
            if start == now:
                state["next"] = now + MIN_GAP
                return
        time.sleep(start - now)


def limited(code: int, body: str) -> None:
    with lock:
        recent.append(True)
        if time.time() < state["pause_until"]:
            return
        state["pauses"] += 1
        if state["pauses"] > 3 or (state["pauses"] > 1 and sum(recent) > 2):
            state["stop"] = True
            print(f"  STOP: rate-limited again after {state['pauses'] - 1} pauses ({sum(recent)}/{len(recent)} recent); quota margin gone", flush=True)
            return
        state["pause_until"] = time.time() + 60
        print(f"  rate-limited ({code} {body[:80]}), pause 60 s (pause {state['pauses']})", flush=True)


def judge(text: str) -> str | None:
    body = json.dumps({"model": "@cf/cloudflare/clef", **flag_request(text)}).encode()
    headers = {"content-type": "application/json", "authorization": f"Bearer {api_key()}", "user-agent": "coach-copilot-revalidate/1.0"}
    for _ in range(6):
        gate()
        try:
            out = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, headers), timeout=60))
        except urllib.error.HTTPError as err:
            msg = err.read().decode(errors="replace")
            if err.code == 429 or "3021" in msg:
                limited(err.code, msg)
            continue
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
            continue
        if "3021" in json.dumps(out.get("error") or ""):
            limited(200, json.dumps(out["error"]))
            continue
        with lock:
            recent.append(False)
        return (((out.get("result") or {}).get("answers") or {}).get("flag") or {}).get("choice")
    return None


def fill(texts: list[str]) -> dict:
    flags = {}
    if os.path.exists(CACHE):
        for row in map(json.loads, open(CACHE)):
            flags[row["text"]] = row["flag"]
    todo = sorted({t for t in texts if flags.get(t) is None})
    print(f"flag version {VERSION}: {len(set(texts))} unique texts, cached {len(set(texts)) - len(todo)}, to judge {len(todo)}, "
          f"{THREADS} in flight, gap {MIN_GAP}s", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(THREADS) as pool, open(CACHE, "a") as cache:
        for n, (text, flag) in enumerate(zip(todo, pool.map(judge, todo)), 1):
            flags[text] = flag
            if flag is not None:
                cache.write(json.dumps({"text": text, "flag": flag}, ensure_ascii=False) + "\n")
            if n % 250 == 0:
                cache.flush()
                print(f"  {n}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    missing = sum(flags.get(t) is None for t in set(texts))
    print(f"done in {time.time() - t0:.0f}s; unanswered {missing}", flush=True)
    return flags


def main() -> None:
    mode = sys.argv[1]
    if mode == "split":
        import pandas as pd
        data, split = sys.argv[2], sys.argv[3]
        lines = pd.read_csv(f"{data}/synthetic_data/synthetic_call_transcripts.csv")
        calls = pd.read_csv(f"{data}/synthetic_data/synthetic_calls.csv")
        calls = calls[(calls.labelled == "yes") & (calls.participant_id.map(split_of) == split)]
        sub = lines[lines.call_id.isin(set(calls.call_id)) & (lines.speaker == "participant") & lines.participant_talk.notna()]
        fill(list(sub.text))
    elif mode == "stress":
        for path in sys.argv[2:]:
            rows = json.load(open(path))["lines"]
            flags = fill([r["text"] for r in rows])
            name = os.path.basename(path).removesuffix(".json")
            out = os.path.join(HERE, "..", "runs", f"{name}-flags-{VERSION}.jsonl")
            with open(out, "w") as f:
                for r in rows:
                    rule, model = red_rule(r["text"]), flags.get(r["text"])
                    f.write(json.dumps({"id": r["id"], "flag": rule[1] if rule else model,
                                        **({"rule": rule[0], "modelFlag": model} if rule else {})}) + "\n")
            print("wrote", os.path.relpath(out, os.path.join(HERE, "..")))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
