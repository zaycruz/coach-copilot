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
7. **Put counter-examples under the label they belong to.** "Not a flag: moods, known side effects, food prices…" written inside `medical` or `dropout` attracted matches. The same examples filed under `none`, plus the everyday barriers that raised false alarms on dev, give the flag results in Evaluation.

Coach conduct is two narrow questions, not one broad one: `routed`, asked only right after a red flag, and `scope`, medical advice or not. One broad "does this coach line need review?" question fired on most coach lines.

The response has, per question, `choice`, `confidence` and `probabilities`. `confidence` is well calibrated (dev split, 120 calls):

| Clef confidence | Share of lines | Accuracy |
|---|---|---|
| ≥ 0.3 | 77% | 98.3% |
| < 0.3 | 23% | 76% → sent to GLM review |

## Evaluation

Labelled synthetic calls are split by `participant_id` hash: **dev 20%** (tuning) and **validation 80%** (held out, reported only). All numbers below are on validation unless marked dev.

### Tags (`scripts/evaluate.py`)

| Validation, 1,085 calls, 56,038 lines | Accuracy |
|---|---|
| Coach move (12-way) | **93.1%** (EN 93.9%, ES 91.6%) |
| Participant talk (3-way) | **90.9%** (EN 91.2%, ES 90.4%) |

Clef latency per line: p50 ~0.3–0.5 s, p95 ~1.05 s. These numbers come from the move and talk questions as they ship. They were not rerun after `scope` joined the coach request, because `scope` is a separate question in the same call.

### Coach score vs supervisors (`scripts/agreement.py`, `GET /api/c5/agreement`)

There are 477 validation calls that both supervisors rated `overall_1_to_5`. The app score is the live screen's arithmetic over Clef's tags. CIs are a bootstrap over calls (2,000 resamples).

| Pair | Pearson r (95% CI) | Spearman ρ (95% CI) |
|---|---|---|
| **App score vs mean of supervisors A and B** | **0.789** (0.754–0.820) | **0.797** (0.758–0.830) |
| App score vs supervisor A | 0.751 (0.708–0.788) | 0.757 (0.711–0.794) |
| App score vs supervisor B | 0.701 (0.657–0.739) | 0.706 (0.655–0.748) |
| Supervisor A vs supervisor B | 0.692 (0.640–0.736) | 0.673 (0.614–0.724) |
| Score from human line labels vs mean of A and B | 0.816 (0.783–0.843) | 0.821 (0.786–0.850) |

The app tracks each supervisor at least as closely as the two supervisors track each other. Supervisor A vs B: ICC(2,1) 0.666 (0.613–0.712), mean absolute difference 0.61 points on the 1–5 scale. Mapped to 1–5 by a least-squares line, the app score is off by 0.55 points on average from supervisor A and 0.60 from B. The score formula itself, run on perfect human labels, reaches r = 0.816, and the app's score correlates r = 0.955 with that. The endpoint returns the per-call table.

### Flags (`evaluate.py --flags-only`, `scripts/flag_eval.py`)

The participant flag is judged in its own line-only request. Its `none` examples come from dev false alarms only.

| | Result | Wilson 95% CI |
|---|---|---|
| `needs_human` lines caught | **13/13** (10 medical, 2 social, 1 distress) | 77.2–100% |
| False alarms, lines | **9 / 25,785** (5 distinct texts, 0 dropout) | 0.02–0.07% |
| False alarms, calls | 9 / 1,085 | 0.4–1.6% |
| Coach conduct vs supervisor `safety_issue_handled` | 8/8 calls | 67.6–100% |
| dev: caught / false alarms | 7/7, 1 / 6,854 lines | |

### Second layer (GLM-5.3-flash)

`scripts/glm_bench.py` compared models on 60 dev lines where Clef's confidence was < 0.3 (Clef alone: 77%):

| Model | Review accuracy | Answered | Review p50 | Cue latency |
|---|---|---|---|---|
| GLM-5.3 | 62–74% | 37–58 / 60 | 3.7–4.2 s | 3–90 s, cold calls up to 48 s |
| **GLM-5.3-flash** | 78–84% | 27–32 / 60 | 2.2–3.1 s | 2–8 s |
| GLM-4.7-flash | 81–86% | 47–51 / 60 | 16–18 s | 7–22 s |

`scripts/review_eval.py` then ran 600 random validation lines with confidence < 0.3 through the live `/api/review`:
- Clef alone: 76.0%. Clef + review: 76.8%.
- The review was right on 91.5% of the lines it answered, against Clef's 80.9% on those same lines.
- It answered only 47 of 600. The other 553 hit the Workers AI per-minute limit (error 3021).

Cues are where GLM earns its place. Review stays an optional background second opinion. Cue language is set by the Worker (`Call language: English|Spanish` from the participant's words); left to infer it, the models sometimes answered in Spanish or French.

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

**Rate limits.** Workers AI caps requests per minute per model (error 3021). The Worker does not retry a rate-limited call, and it answers `429` with `retry-after: 20`. The UI keeps Clef's tag when a review gets no label. Batch scripts against the live Worker share this quota with live calls and with Patient 360, so do not run them during a demo.

## Patient 360 endpoints

`src/c5.ts` serves the suite endpoints that Patient 360 (coach-console) asks for in its `INTEGRATION.md`. They use the same shape as the console's adapter, so the console switched over with no change (`x-c5-source: native`):

- `GET /api/c5/calls[/:callId]`: the 27 supervisor-labelled demo calls, scored with the live screen's arithmetic.
- `GET /api/c5/survivors[/:survivorId]`: one row per participant (`c5-P0540`), with status from their worst call.
- `GET /api/c5/demo`: held-out safety call CALL05423, pre-played through the live engine (`scripts/preplay.py`). It has every line's Clef tag and flag, the red medical banner at 05:12, 4 GLM cues, and score 74 (82 on the supervisor labels). 90.6% of its tags match the human labels. A console can open on it with no model calls.
- `GET /api/c5/agreement`: the supervisor-agreement figures above, with the per-call table (app score, human-label score, supervisor A, supervisor B) and the flag Wilson CIs.

The field-level contract, with real responses, is in [`CONTRACT.md`](CONTRACT.md) (`scripts/contract.py`).

The flag kind (medical, social…) on each `needs_human` line is Clef's answer to the line-only flag question, stored in `public/demo-calls.json`. The console's adapter guessed it from keywords; the answers are the same on all 9 lines.

## Sources

- AnnoMI: Wu Z, et al. *Creation, Analysis and Evaluation of AnnoMI, a Dataset of Expert-Annotated Counselling Dialogues.* Future Internet, 2023.
- Text coaching corpus: Gupta I, et al. *Human-Human Health Coaching via Text Messages: Corpus, Annotation, and Analysis.* SIGDIAL 2020.
- Synthetic data: Crane Lab, Sylvester Comprehensive Cancer Center.
