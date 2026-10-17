"""Compare runs/metrics.json with the README headline numbers in scripts/expected.json.
Each value is rounded to the precision the README shows. Prints a table, exits 1 on any mismatch.
Usage: check_metrics.py [metrics.json] [expected.json]"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
got = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "runs", "metrics.json")))
want = json.load(open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "expected.json")))



def dig(obj, path):
    # Keys may contain dots ("shareAtLeast0.3"); try the longest key first at each level.
    parts = path.split(".")
    while parts:
        if isinstance(obj, list):
            obj, parts = obj[int(parts[0])], parts[1:]
            continue
        for n in range(len(parts), 0, -1):
            key = ".".join(parts[:n])
            if key in obj:
                obj, parts = obj[key], parts[n:]
                break
        else:
            raise KeyError(path)
    return obj


bad = 0
print("| README number | metrics.json key | expected | got | ok |\n|---|---|---|---|---|")
for key, exp in want.items():
    if key.startswith("_"):
        continue
    val = dig(got, key)
    places = len(str(exp).split(".")[1]) if isinstance(exp, float) else 0
    ok = round(float(val), places) == round(float(exp), places)
    bad += not ok
    print(f"| {exp} | `{key}` | {exp} | {round(float(val), places) if places else val} | {'yes' if ok else '**NO**'} |")
print(f"\n{len(want) - 1 - bad}/{len(want) - 1} match")
sys.exit(1 if bad else 0)
