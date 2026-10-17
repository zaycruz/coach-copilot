"""The model-free red-flag screen from src/redflags.json, as the Worker runs it: patterns over lowercased, accent-stripped text."""
import json, os, re, unicodedata

RULES = [(r["id"], r["kind"], re.compile(r["pattern"]))
         for r in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "redflags.json")))["rules"]]


def fold(text):
    return "".join(c for c in unicodedata.normalize("NFD", str(text).lower()) if not unicodedata.category(c).startswith("M"))


def red_rule(text):
    """(rule id, kind) of the first matching rule, or None."""
    t = fold(text)
    return next(((i, k) for i, k, p in RULES if p.search(t)), None)
