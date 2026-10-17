"""The coach score: the live screen's MITI-style arithmetic (public/index.html updateScore, src/c5.ts scoreMoves)."""

SUPPORT = {"affirm", "seek collaboration", "emphasize autonomy", "ask permission"}


def score(moves):
    if len(moves) < 4:
        return None
    n = lambda t: sum(m == t for m in moves)
    oq, cq, sr, cr = n("open question"), n("closed question"), n("simple reflection"), n("complex reflection")
    ad, na = sum(m in SUPPORT for m in moves), n("persuade") + n("confront")
    q, r = oq + cq, sr + cr
    return (round(30 * min((r / q if q else (2 if r else 0)) / 2, 1)) + round(20 * min((cr / r if r else 0) / 0.4, 1))
            + round(20 * min((oq / q if q else 0) / 0.5, 1)) + round(30 * (ad / (ad + na) if ad + na else 1)))
