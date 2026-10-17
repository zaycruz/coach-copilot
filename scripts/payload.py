"""Clef payload builders for one turn. Each question is a single decision with
mutually exclusive answers; decision rules sit in the instructions, and each
criterion says when to pick it and what it is not."""

# Precedence is the order of the rules: the first rule that fits wins.
COACH_RULES = (
    "Classify ONLY the line in `line`, using `previous` for context. Apply these rules in order and stop at the first that fits: "
    "0) Call opening or closing small talk: greeting, 'is now a good time?', 'how's your Tuesday going?', next contact ('same time next Tuesday?', 'I'll text you midweek'), goodbye -> other. "
    "1) Criticises, doubts or argues with the participant -> confront. "
    "2) Tells the participant what to do, sets their goal for them, or warns of consequences -> persuade. "
    "3) A statement about the participant's own words, feelings, wants, values, thinking or habits ('You want...', 'You're starting to see...', 'Once you decide something, you follow through') -> simple reflection or complex reflection, never affirm or give information, even if it is positive or ends with 'right?'. "
    "4) Asks before sharing advice -> ask permission. "
    "5) Otherwise asks a question: if it can be answered with yes, no, a number or one word -> closed question ('Any problems this week?', 'Could you do them while you watch TV?'); else open question. "
    "6) Otherwise pick the label whose description fits best."
)

COACH_MOVES = {
    "open question": "A question that invites a long answer: what, how, why, tell me. Not a yes/no question, not a scale question.",
    "closed question": "A question answered with yes, no, a number or one fact, including 'on a scale of 0 to 10'. Not a restatement.",
    "simple reflection": "Paraphrases ONE thing the participant said: how the week went, a feeling, a doubt, or their plan ('It's been a mixed week', 'You're not sure it's worth it', 'Your family is a big part of why', 'You've got a plan in mind'). Adds no preference, trait or reason they did not state.",
    "complex reflection": "Goes beyond their words: names a preference, trait, value or underlying reason they did not state ('You'd rather start small and actually stick with it', 'Once you decide something, you follow through', 'You want to be around for the people you love', 'Once the routine broke, it was easy to lose track'), names ambivalence ('part of you'), or summarises several things ('Let me make sure I got it...', 'So, to sum up...').",
    "affirm": "Directly praises an effort or achievement ('That took effort', 'Good for you', 'That's real progress'). Not a description of what the participant wants or thinks, not 'okay, good' or 'great to hear', not a goodbye.",
    "seek collaboration": "Proposes or agrees an agenda, or invites the participant to shape the plan together ('anything you want to add?').",
    "emphasize autonomy": "Says the decision belongs to the participant ('it's your call', 'you know your schedule best').",
    "ask permission": "Asks whether it is okay to share an idea, advice or information. Does not contain the advice itself.",
    "give information": "States a neutral fact or tip, without telling this participant to do it.",
    "persuade": "Tells the participant what to do, sets the goal for them, or warns them ('next week I want you to', 'you need to', 'you can't be eating at 10', 'if you don't do this, your health will suffer').",
    "confront": "Criticises, doubts or argues with the person ('that sounds like an excuse', 'come on, you can find 10 minutes', 'you said that last week', 'you're not really trying').",
    "other": "Greetings, goodbyes (including 'take care, you're doing great'), short acknowledgements ('okay', 'okay, good', 'I see', 'that's great to hear'), logistics and scheduling.",
}

PARTICIPANT_RULES = (
    "Classify ONLY the line in `line`, spoken by the participant, about the behaviour goal being discussed. "
    "If the line contains both reasons to change and reasons not to, pick the one that ends the line."
)

PARTICIPANT_TALK = {
    "change": "Desire, ability, reasons, need or commitment to change, a plan or backup plan, or steps already taken ('I can probably manage three days', 'if it rains I'll walk in the hallway', 'I wrote it on the calendar').",
    "sustain": "Reasons, barriers, habits or excuses for not changing, putting it off, or pushing back ('I forgot after Tuesday', 'I don't know where the day goes', 'everybody gets home late, so dinner is late', 'I've always eaten late', 'walking hurts my knees, what for?', 'maybe after the summer', 'easy for you to say').",
    "neutral": "Agreeing with or confirming what the coach said ('perfect', 'that works', 'yes, please', 'that's it exactly', 'claro'), short answers ('no, not really', 'not yet'), greetings, scheduling and small talk. These give no reason for or against the change.",
}

FLAG = {
    "none": "Nothing anyone outside the call must act on today. This includes ordinary barriers and moods ('busy', 'tired', 'rough week', 'I haven't been sleeping', 'kind of down', 'I'm just so tired'), doubts about the goal ('I don't really see the point', 'walking hurts my knees, so why bother'), known ongoing side effects ('my feet are numb again', 'no energy after treatment', 'the hot flashes are back'), everyday aches ('my back is acting up', 'my knees have been bothering me'), prices of healthy food ('vegetables are so expensive'), hunger at night ('by 10 o'clock I'm starving'), a neighbourhood that feels unsafe for walking ('I don't feel safe walking around here'), disliking the goal ('water just doesn't do it for me'), reasons and doubts about changing ('I need this for my head too, not just my body', 'I don't want the cancer back', 'I've tried this before and it never lasts'), agreeing or thanking ('okay', 'sí, está bien', 'thank you'), and skipping or rescheduling the goal or call.",
    "medical": "A new or worsening symptom that needs the care team today: fever during or after treatment, chest pain, shortness of breath, a fall, a swollen, red or warm leg, diarrhoea or vomiting for days, bleeding, new severe pain.",
    "distress": "Suicide risk, said directly or hinted: hopelessness about life itself ('what's the point of any of this', 'why bother with anything anymore'), not wanting to live or wake up, wishing to be dead, feeling a burden or that others would be better off; and warning signs, even said calmly or as a joke: saying goodbye or leaving messages 'in case', giving possessions or pets away, putting affairs in order, counting, saving or asking about extra pills, having a rope, gun or other means, having a plan, refusing all further treatment because nothing matters.",
    "social": "A basic need is not met: days without enough food, can't afford medicine or rent, no way to get to treatment, unsafe at home.",
    "dropout": "Wants to quit, pause or stop the coaching program or the calls, or says the program is not helping ('I want to stop the calls', 'this isn't working for me', 'I'm thinking of leaving the program').",
}

FLAG_INSTRUCTIONS = "This line was said by a participant on a health-coaching call. Does it raise something someone outside the call must act on today?"

ROUTED = {
    "routed": "The coach acknowledges the concern AND says the care team, a doctor, a nurse or a navigator will be told or should be contacted, or asks about safety ('Thank you for telling me. Your care team needs to hear this today').",
    "not_routed": "The coach does not route it: a bare acknowledgement ('Right.', 'Ok.', 'Entiendo.', 'I see.'), sympathy without a next step, or moving on to the goal.",
}

ROUTED_INSTRUCTIONS = "In `previous`, the participant raised a concern that needs follow-up today (a symptom, hopelessness or an unmet basic need). Judge ONLY the coach's reply in `line`."

SCOPE = {
    "none": "Within a health coach's role. This includes lifestyle advice and opinions about food, drinks, sleep or exercise, even pushy ones ('you should stop buying soda', 'you need to walk every day'), and telling them to call their care team.",
    "medical_advice": "Outside a health coach's role: diagnoses a symptom, tells them to start, stop or change a medicine, dose or treatment, or says a symptom is nothing to worry about.",
}

SCOPE_INSTRUCTIONS = "Judge ONLY the coach's line in `line`. Is it within a health coach's role?"


def turn_state(line: dict, previous: list[dict]) -> dict:
    """The target line apart from its context, so no question is unclear about which line it judges."""
    return {
        "setting": "Phone call in a health-coaching program for cancer survivors. English or Spanish.",
        "previous": [f"{p['speaker']}: {p['text']}" for p in previous],
        "line": {"speaker": line["speaker"], "text": line["text"]},
    }


def turn_questions(speaker: str) -> dict:
    """Questions that share the turn state (line + previous lines)."""
    if speaker == "coach":
        return {
            "move": {"type": "choice", "instructions": COACH_RULES, "criteria": COACH_MOVES},
            "scope": {"type": "choice", "instructions": SCOPE_INSTRUCTIONS, "criteria": SCOPE},
        }
    return {"talk": {"type": "choice", "instructions": PARTICIPANT_RULES, "criteria": PARTICIPANT_TALK}}


def flag_request(text: str) -> dict:
    """The participant flag is its own request with the line alone, so an earlier concern cannot leak into it."""
    return {"state": {"line": {"speaker": "participant", "text": text}},
            "questions": {"flag": {"type": "choice", "instructions": FLAG_INSTRUCTIONS, "criteria": FLAG}}}


def routed_question() -> dict:
    """Asked on the coach line right after a flagged participant line, in the same request as `move`."""
    return {"routed": {"type": "choice", "instructions": ROUTED_INSTRUCTIONS, "criteria": ROUTED}}


if __name__ == "__main__":
    import json, sys
    json.dump({"coach_rules": COACH_RULES, "coach_moves": COACH_MOVES, "participant_rules": PARTICIPANT_RULES,
               "participant_talk": PARTICIPANT_TALK, "flag": FLAG, "flag_instructions": FLAG_INSTRUCTIONS,
               "routed": ROUTED, "routed_instructions": ROUTED_INSTRUCTIONS, "scope": SCOPE, "scope_instructions": SCOPE_INSTRUCTIONS},
              open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
