"""Per-turn Clef tagging over the labelled synthetic calls.

Split: labelled calls are assigned to dev (20%) or validation (80%) by a hash of
participant_id, so one person's calls never straddle the split. Tune on dev,
report on validation only.

The participant flag is its own line-only request (payload.flag_request), like the Worker, so it is judged
once per unique participant text and cached apart from the tags.

Usage: evaluate.py <data_dir> <split dev|validation> [max_calls] [model] [context_lines] [--flags-only]
Writes runs/<split>-<model>.jsonl (one row per line, cached per unique turn) and prints metrics.
--flags-only skips the tag requests and writes runs/<split>-<model>-flags.jsonl (participant lines only).
"""
import hashlib, json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from payload import turn_state, turn_questions, flag_request, FLAG, FLAG_INSTRUCTIONS

RED = {"medical", "distress", "social"}

URL = "https://coach-copilot-clef.zay-259.workers.dev/judge"
HERE = os.path.dirname(os.path.abspath(__file__))
KEY = open(os.path.join(HERE, "..", ".env")).read().split("CLEF_API_KEY=", 1)[1].split()[0]


def split_of(participant_id: str) -> str:
    return "dev" if int(hashlib.sha1(participant_id.encode()).hexdigest(), 16) % 5 == 0 else "validation"


def post(payload: dict) -> dict:
    body = json.dumps(payload).encode()
    headers = {"content-type": "application/json", "authorization": f"Bearer {KEY}", "user-agent": "coach-copilot/0.1"}
    for wait in (1, 2, 4, 8, 0):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(URL, body, headers), timeout=60))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as err:
            if not wait:
                return {"error": str(err)}
            time.sleep(wait)


def main() -> None:
    flags_only = "--flags-only" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--flags-only"]
    data, split = sys.argv[1], sys.argv[2]
    max_calls = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    model = sys.argv[4] if len(sys.argv) > 4 else "@cf/cloudflare/clef"
    ctx = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    lines = pd.read_csv(f"{data}/synthetic_data/synthetic_call_transcripts.csv")
    calls = pd.read_csv(f"{data}/synthetic_data/synthetic_calls.csv")
    calls = calls[calls.labelled == "yes"].copy()
    calls["split"] = calls.participant_id.map(split_of)
    ids = calls[calls.split == split].call_id
    if max_calls:
        ids = ids.sample(min(max_calls, len(ids)), random_state=7)
    sub = lines[lines.call_id.isin(set(ids))].sort_values(["call_id", "line_number"])

    turns = []
    for _, call in sub.groupby("call_id"):
        rows = call.to_dict("records")
        for i, r in enumerate(rows):
            gold = r["coach_move"] if r["speaker"] == "coach" else r["participant_talk"]
            if not isinstance(gold, str):
                continue
            prev = rows[max(0, i - ctx):i]
            state = turn_state(r, prev)
            key = json.dumps(state, sort_keys=True, ensure_ascii=False)
            turns.append({"call_id": r["call_id"], "line_number": r["line_number"], "speaker": r["speaker"], "text": r["text"],
                          "gold": gold, "gold_safety": r.get("needs_human"), "key": key, "state": state})
    unique = {t["key"]: t for t in turns}
    print(f"{split}: {len(ids)} calls, {len(turns)} labelled turns, {len(unique)} unique turn states", flush=True)

    def judge(t):
        t0 = time.time()
        out = post({"model": model, "state": t["state"], "questions": turn_questions(t["speaker"])})
        return t["key"], out, time.time() - t0

    os.makedirs(os.path.join(HERE, "..", "runs"), exist_ok=True)
    version = hashlib.sha1(json.dumps([turn_questions("coach"), turn_questions("participant")], sort_keys=True).encode()).hexdigest()[:8]
    cache_path = os.path.join(HERE, "..", "runs", f"cache-{model.split('/')[-1]}-ctx{ctx}-{version}.jsonl")
    answers, lat = {}, []
    if os.path.exists(cache_path):
        for row in map(json.loads, open(cache_path)):
            if "error" not in row["out"]:
                answers[row["key"]] = row["out"]
    todo = [] if flags_only else [t for k, t in unique.items() if k not in answers]
    if not flags_only:
        print(f"  cached {len(unique) - len(todo)}, to judge {len(todo)}", flush=True)
    with ThreadPoolExecutor(32) as pool, open(cache_path, "a") as cache:
        for n, (key, out, secs) in enumerate(pool.map(judge, todo), 1):
            answers[key] = out
            lat.append(secs)
            cache.write(json.dumps({"key": key, "out": out}, ensure_ascii=False) + "\n")
            if n % 1000 == 0:
                cache.flush()
                print(f"  {n}/{len(todo)}", flush=True)
    lat = lat or [0.0]
    flags = judge_flags(model, sorted({t["text"] for t in turns if t["speaker"] == "participant"}))

    tag = model.split("/")[-1]
    path = os.path.join(HERE, "..", "runs", f"{split}-{tag}-ctx{ctx}{'-flags' if flags_only else ''}.jsonl")
    hits, errors, confusion = Counter(), 0, Counter()
    with open(path, "w") as f:
        for t in turns:
            if flags_only and t["speaker"] == "coach":
                continue
            flag = flags.get(t["text"]) if t["speaker"] == "participant" else None
            row = {k: t[k] for k in ("call_id", "line_number", "speaker", "text", "gold", "gold_safety")} | {"flag": flag}
            if not flags_only:
                out = answers.get(t["key"], {"error": "missing"})
                q = "move" if t["speaker"] == "coach" else "talk"
                a = ((out.get("result") or {}).get("answers") or {}).get(q) or {}
                pred = a.get("choice")
                errors += "error" in out
                hits[(q, "n")] += 1
                hits[(q, "hit")] += pred == t["gold"]
                if pred != t["gold"]:
                    confusion[(q, t["gold"], pred)] += 1
                row |= {"pred": pred, "confidence": a.get("confidence"), "probabilities": a.get("probabilities")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    lat.sort()
    if not flags_only:
        print(f"model={model} ctx={ctx} errors={errors} p50={lat[len(lat)//2]:.2f}s p95={lat[int(len(lat)*.95)]:.2f}s")
        for q in ("move", "talk"):
            if hits[(q, "n")]:
                print(f"  {q}: {hits[(q, 'hit')]}/{hits[(q, 'n')]} = {hits[(q, 'hit')] / hits[(q, 'n')]:.1%}")
    flag_metrics([t for t in turns if t["speaker"] == "participant"], flags)
    for (q, g, p), n in confusion.most_common(10):
        print(f"    {n:4d} {q}: gold={g} pred={p}")
    print("wrote", path)


def judge_flags(model: str, texts: list[str]) -> dict:
    """One line-only flag request per unique participant text, cached by flag-question version."""
    version = hashlib.sha1(json.dumps([FLAG, FLAG_INSTRUCTIONS], sort_keys=True).encode()).hexdigest()[:8]
    cache_path = os.path.join(HERE, "..", "runs", f"cache-flag-{model.split('/')[-1]}-{version}.jsonl")
    flags = {}
    if os.path.exists(cache_path):
        for row in map(json.loads, open(cache_path)):
            flags[row["text"]] = row["flag"]
    todo = [t for t in texts if flags.get(t) is None]
    print(f"flags: {len(texts)} unique participant texts, cached {len(texts) - len(todo)}, to judge {len(todo)}", flush=True)

    def one(text):
        out = post({"model": model, **flag_request(text)})
        return text, (((out.get("result") or {}).get("answers") or {}).get("flag") or {}).get("choice")

    with ThreadPoolExecutor(32) as pool, open(cache_path, "a") as cache:
        for n, (text, flag) in enumerate(pool.map(one, todo), 1):
            flags[text] = flag
            if flag is not None:
                cache.write(json.dumps({"text": text, "flag": flag}, ensure_ascii=False) + "\n")
            if n % 1000 == 0:
                cache.flush()
                print(f"  {n}/{len(todo)}", flush=True)
    return flags


def flag_metrics(lines: list[dict], flags: dict) -> None:
    """Caught = needs_human lines with any flag. False alarm = any flag on a line not marked needs_human.
    needs_human is gold for red flags only, so dropout is a firing rate, not a false-alarm count."""
    missing = sum(flags.get(t["text"]) is None for t in lines)
    true = [t for t in lines if t["gold_safety"] == "yes"]
    other = [t for t in lines if t["gold_safety"] != "yes"]
    fired = lambda t, kinds=None: (flags.get(t["text"]) or "none") != "none" and (kinds is None or flags[t["text"]] in kinds)
    print(f"  flag: no answer {missing}/{len(lines)} lines")
    print(f"  caught {sum(map(fired, true))}/{len(true)}:", dict(Counter(flags.get(t['text']) for t in true)))
    for name, kinds in (("any flag", None), ("red", RED), ("dropout", {"dropout"})):
        hit = [t for t in other if fired(t, kinds)]
        calls = {t["call_id"] for t in hit}
        print(f"  {name} on other lines: {len(hit)}/{len(other)} lines, {len({t['text'] for t in hit})} unique texts, "
              f"{len(calls)}/{len({t['call_id'] for t in lines})} calls")
    for text, n in Counter(t["text"] for t in other if fired(t)).most_common(15):
        print(f"    {n:4d} {flags[text]:9s} {text[:90]}")


if __name__ == "__main__":
    main()
