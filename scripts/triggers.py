"""When the page asks GLM for a cue: public/index.html checkTriggers, line for line (also used by preplay.py and metrics.py)."""
FAMILY = {"open question": "q", "closed question": "q", "simple reflection": "r", "complex reflection": "r"}
CUE_GAP = 3
FLAG_TRIGGERS = {
    "missed_concern": "The participant raised a concern and the coach moved on without routing it. Help the coach go back to it now.",
    "medical_advice": "The coach just gave medical advice outside their role. Help the coach walk it back and refer to the care team.",
    "dropout": "The participant is talking about leaving the program. Help the coach explore it without pressure.",
}
COACHING = {"confront": "The coach just confronted the participant.",
            "persuade": "The coach just told the participant what to do without asking permission.",
            "unreflected_change": "The participant voiced change talk and the coach did not reflect or affirm it.",
            "four_questions": "The coach has asked four questions in a row without a reflection.",
            "three_closed": "Three closed questions in a row."}


def coaching_kind(lines, i):
    """Which coaching trigger line i meets, ignoring the cue gap; None if none. lines: dicts with speaker, tag."""
    l = lines[i]
    if l["speaker"] != "coach":
        return None
    coach = [x for x in lines[:i + 1] if x["speaker"] == "coach" and x["tag"]]
    last, before = [x["tag"] for x in coach[-4:]], lines[i - 1] if i else None
    if l["tag"] == "confront":
        return "confront"
    if l["tag"] == "persuade" and not any(x["tag"] == "ask permission" for x in coach[-3:-1]):
        return "persuade"
    if before and before["speaker"] == "participant" and before["tag"] == "change" and FAMILY.get(l["tag"]) != "r" and l["tag"] != "affirm":
        return "unreflected_change"
    if len(last) == 4 and all(FAMILY.get(x) == "q" for x in last):
        return "four_questions"
    if len(last) >= 3 and all(x == "closed question" for x in last[-3:]):
        return "three_closed"
    return None


def trigger_for(lines, i, last_cue_at):
    """(trigger text, kind) for line i, or (None, None). lines also carry flag."""
    l = lines[i]
    flag = l.get("flag") or "none"
    if flag in FLAG_TRIGGERS:
        return FLAG_TRIGGERS[flag], flag
    if flag != "none":
        return f"Participant raised a {flag} concern. Help the coach respond with care and connect them to the right person today.", "red_flag"
    kind = coaching_kind(lines, i)
    if kind and sum(x["speaker"] == "coach" for x in lines[:i + 1]) - last_cue_at < CUE_GAP:
        return None, None
    return (COACHING[kind], kind) if kind else (None, None)


def play(lines):
    """Run a whole call; returns [(line index, kind, trigger text)] for every cue the page would ask for."""
    out, last_cue_at = [], -99
    for i in range(len(lines)):
        text, kind = trigger_for(lines, i, last_cue_at)
        if text:
            last_cue_at = sum(x["speaker"] == "coach" for x in lines[:i + 1])
            out.append((i, kind, text))
    return out
