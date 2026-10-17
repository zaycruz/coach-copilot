# Coach's Co-pilot

Emerge Hackathon · Sylvester healthcare track · Challenge 5 ("Follow and score a health-coaching conversation").

Follows a coaching call live or recorded, spoken or typed. Tags every line with a motivational-interviewing (MI) code, scores the coach, raises a red banner for anything a person must act on today, and gives the coach a cue only when one is needed.

Live: https://coach.raava.dev

## Architecture

```
speech (Safari = Apple speech engine) ─┐
recorded / typed lines ────────────────┼─> Worker (coach.raava.dev) ──> AI Gateway "coach-copilot" ──> Workers AI
                                       │     /api/turn   every line        Clef      (~0.3–0.5 s)
                                       │     /api/review conf < 0.3 only    GLM-5.3-flash (async, ~2–4 s)
                                       │     /api/cue    triggers only      GLM-5.3-flash (async, ~2–5 s)
browser: score, banner, cues <─────────┘
```

- **Clef** (`@cf/cloudflare/clef`) classifies. One request per line, with the two previous lines as context. Coach lines get `move` (12 labels) and `scope`, plus `routed` right after a flagged participant line. Participant lines get `talk` (change/sustain/neutral). A second, parallel Clef request judges the participant `flag` (see Flags).
- **GLM-5.3-flash** (`@cf/zai-org/glm-5.3-flash`) runs only when needed and never blocks the tag. It won the bake-off against GLM-5.3 and GLM-4.7-flash (see Evaluation). Requests may pass `model` to try the others.
  - *Review*: when Clef's confidence is below 0.3 (about 23% of lines), GLM picks among Clef's top 4 labels and may change the tag.
  - *Cues*: rule triggers (confront, unasked-for persuasion, missed change talk, four questions without a reflection, three closed questions in a row, any flag). There is a cooldown of three coach lines between cues. The screen stays quiet otherwise.
- **Score** is arithmetic over the tags, using MITI-style targets: 2 reflections per question (30 pts), 40% complex reflections (20), 50% open questions (20), and MI-consistent vs. persuade/confront moves (30).
- **Flags** answer the brief's "flag anything a person should see". There is no human review queue in the pipeline; the banner is the output. Up to three banners stack:

  | Flag | Banner | Raised when |
  |---|---|---|
  | `medical` | red | participant reports a new or worsening symptom: care team today |
  | `distress` | red | hopelessness about life itself, not ordinary low mood |
  | `social` | red | food, money, transport or safety need |
  | `dropout` | amber | participant signals they may quit the program |
  | `missed_concern` | amber | coach's next line after a red flag does not acknowledge it or route it (coach conduct) |
  | `medical_advice` | amber | coach gives medical advice (dosing, diagnosis, stopping treatment); lifestyle advice is fine |

## Clef payload

The source of truth is `scripts/payload.py`. Running `python3 scripts/payload.py src/payload.json` exports it for the Worker.

Design rules for every question:

1. **One decision per question**, with mutually exclusive answers.
2. **Precedence rules in `instructions`** settle overlaps. The first rule that fits wins. Example: a positive "You want to…" statement is a reflection, never an affirm.
3. **Every criterion says when to pick it, with examples.** The examples are the exact phrasings Clef confused on the dev split, filed under the label they belong to (rule 7).
4. **The target line sits apart from its context** (`line` vs. `previous`), so no question is unclear about which line it judges.
5. **Short context** (2 previous lines) keeps requests fast and is enough to tell a reflection from a statement.

```json
{
  "model": "@cf/cloudflare/clef",
  "state": {
    "setting": "Phone call in a health-coaching program for cancer survivors. English or Spanish.",
    "previous": [
      "coach: What happened with the vegetables?",
      "participant: Honestly the prices. Fresh stuff is expensive."
    ],
    "line": { "speaker": "coach", "text": "You want to eat better, and the prices make it feel out of reach." }
  },
  "questions": {
    "move": {
      "type": "choice",
      "instructions": "Classify ONLY the line in `line`, using `previous` for context. Apply these rules in order and stop at the first that fits: 0) opening/closing small talk -> other. 1) criticises, doubts or argues -> confront. 2) tells them what to do, sets their goal, or warns -> persuade. 3) a statement about the participant's own words, feelings, wants, values, thinking or habits -> simple/complex reflection, never affirm or give information. 4) asks before sharing advice -> ask permission. 5) a question: yes/no/number/one word -> closed, else open. 6) otherwise the best-fitting label.",
      "criteria": {
        "open question": "…", "closed question": "…",
        "simple reflection": "Paraphrases ONE thing the participant said… Adds no preference, trait or reason they did not state.",
        "complex reflection": "Goes beyond their words: names a preference, trait, value or underlying reason…, names ambivalence, or summarises several things.",
        "affirm": "…", "seek collaboration": "…", "emphasize autonomy": "…", "ask permission": "…",
        "give information": "…", "persuade": "…", "confront": "…", "other": "…"
      }
    }
  }
}
```

Two more rules came from the flag work:

6. **Judge a flag on the line alone.** The participant `flag` runs as a separate request whose state is only `{line}`, in parallel with the main one, so it costs no latency. With context in the same request, an earlier concern leaked forward ("Okay, thank you." was flagged medical).
7. **Put counter-examples under the label they belong to.** "Not a flag: moods, known side effects, food prices…" written inside `medical` or `dropout` attracted matches. Moved into `none`, the same examples cut false alarms on the earlier list from 27 texts to 3, with 13/13 true cases still caught.

Coach conduct is two narrow questions, not one broad one. A single "does this coach line need review?" flag fired on most coach lines (3/8 calls agreed with supervisors). `routed` (asked only right after a red flag) and `scope` (medical advice or not) agree on 6/8.

The response has, per question, `choice`, `confidence` and `probabilities`. `confidence` is well calibrated (dev split, 120 calls):

| Clef confidence | Share of lines | Accuracy |
|---|---|---|
| ≥ 0.3 | 77% | 98.3% |
| < 0.3 | 23% | 76% → sent to GLM review |

## Evaluation

Labelled synthetic calls are split by `participant_id` hash: **dev 20%** (tuning) and **validation 80%** (held out, reported only). See `scripts/evaluate.py`; per-line outputs go to `runs/`.

| Split | Coach move (12-way) | Participant talk (3-way) | Flags |
|---|---|---|---|
| dev, first payload (40 calls) | 79.5% | 86.6% | – |
| dev, final payload (120 calls) | 93.4% | 92.6% | 2/2 caught, 2 false alarms |
| **validation (1,085 calls, 56,001 lines)** | **93.1%** (EN 93.9%, ES 91.6%) | **90.9%** (EN 91.2%, ES 90.4%) | **13/13 caught, 9 false alarms / 25,785 lines** (9 calls) |

Flags are judged in their own line-only request (`evaluate.py --flags-only`, one request per unique participant text):

| Flag payload | Split | Caught | False alarms (any flag, lines not `needs_human`) |
|---|---|---|---|
| before redesign (flag in main request) | validation | 13/13 | 71 / 25,781 |
| line-only request | dev | 7/7 | 12 / 6,854 |
| line-only request | validation | 13/13 | 68 / 25,785 (30 texts, 66 calls) |
| **+ dev counter-examples in `none`** (aches, night hunger, unsafe-to-walk area, disliking the goal) | dev | 7/7 | 1 / 6,854 |
| **+ dev counter-examples in `none`** | **validation** | **13/13** | **9 / 25,785** (5 texts, 9 calls), 0 dropout |

The counter-examples were taken from dev false alarms only. Replayed line by line against the live Worker (`scripts/flag_eval.py`), 1 of 27 earlier false-alarm texts is still flagged, and coach conduct agrees with supervisors on **8/8** calls. Move and talk accuracy were not rerun after `scope` joined the coach request.

Clef latency per line: p50 ~0.3–0.5 s, p95 ~1.05 s.

### Second layer bake-off

`scripts/glm_bench.py`, 60 dev lines where Clef's confidence was < 0.3 (Clef alone: 77%). Review accuracy is over the lines each model answered; unanswered lines were upstream 502s (now retried once) or no valid label.

| Model | Review accuracy | Answered | Review p50 | Cue latency |
|---|---|---|---|---|
| GLM-5.3 | 62–74% | 37–58 / 60 | 3.7–4.2 s | 3–90 s, cold calls up to 48 s |
| **GLM-5.3-flash** | 78–84% | 27–32 / 60 | 2.2–3.1 s | 2–8 s |
| GLM-4.7-flash | 81–86% | 47–51 / 60 | 16–18 s | 7–22 s |

Ranges are two runs. GLM review is roughly a wash against Clef on its own uncertain lines, so it stays a background second opinion. Cues are where GLM earns its place, and flash writes them as well as GLM-5.3 at a fraction of the wait. Cue language is set by the Worker (`Call language: English|Spanish` from the participant's words); left to infer it, the models sometimes answered in Spanish or French.

Validation check (`scripts/review_eval.py`, 600 random validation lines with confidence < 0.3, live `/api/review`): Clef alone 76.0%, Clef + review 76.7%. The review fixed 8 and broke 4, but it answered only 65/600 lines (p50 4.3 s); the rest kept Clef's tag. The review stays an optional background second opinion; its low answer rate under batch load is not yet diagnosed.

### AnnoMI expert benchmark

`scripts/annomi_bench.py`: the 428 lines labelled by 10 experts (7 real counselling sessions), shipped payload unchanged, 2 previous lines of context. Clef's 12 moves are collapsed to AnnoMI's 4: questions and *ask permission* → question; simple/complex reflection → reflection; give information and persuade → information or advice; the rest → other.

| Lines | Clef | Experts agree with the majority label |
|---|---|---|
| counsellor (216) | 75.0% | 87.5% |
| client talk (212) | 75.0% | 80.2% |
| **all 428** | **75.0%** | **83.9%** |

Where all 10 experts agreed, Clef gets 91.1% (counsellor, 112 lines) and 93.1% (client, 58). Where experts split (≤ 70% agreement), Clef gets 50–58%. Most misses are client change or sustain talk read as neutral (37 lines). Subtypes, scored where the coarse label matched: open/closed 73.2% (56), simple/complex 39.1% (23). The payload was tuned on synthetic cancer-coaching calls, not on AnnoMI.

## Run

```sh
npm install
npx wrangler deploy                                   # coach.raava.dev + workers.dev
uv run --with pandas python -I scripts/evaluate.py <data_dir> dev 120
uv run --with pandas python -I scripts/evaluate.py <data_dir> validation 0 @cf/cloudflare/clef 2 --flags-only
uv run --with pandas python -I scripts/annomi_bench.py <data_dir>
```

The `/judge` endpoint is a raw Clef proxy for batch scripts (`Authorization: Bearer $CLEF_API_KEY` from `.env`).

## Sources

- AnnoMI: Wu Z, et al. *Creation, Analysis and Evaluation of AnnoMI, a Dataset of Expert-Annotated Counselling Dialogues.* Future Internet, 2023.
- Text coaching corpus: Gupta I, et al. *Human-Human Health Coaching via Text Messages: Corpus, Annotation, and Analysis.* SIGDIAL 2020.
- Synthetic data: Crane Lab, Sylvester Comprehensive Cancer Center.
