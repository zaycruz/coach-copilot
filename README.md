# Coach's Co-pilot

Emerge Hackathon · Sylvester healthcare track · Challenge 5. The engine and API behind the coaching view in Patient 360.

**The app's coach score agrees with each supervisor about as well as the two supervisors agree with each other: r = 0.75 with supervisor A and 0.70 with supervisor B, against 0.69 between A and B.**
For the care team, every coaching call can be scored and screened for urgent concerns as it happens, so supervisors no longer hand-review a sample.

| Result (held-out validation split) | Baseline | 95% CI |
|---|---|---|
| Coach score vs each supervisor: **r = 0.751** (A), **0.701** (B), 477 calls | Supervisor B vs A: r = 0.692 | 0.708–0.788 (A), 0.657–0.739 (B) |
| Coach move tags: **93.1%** correct (12 labels, 1,085 calls) | Always the most common label: 20.3% | 92.8–93.4% |
| Urgent lines caught: **13 / 13**, with 244 false alarms in 25,785 lines (9 before the suicide-risk fix) | Base rate: 0.05% of lines are urgent | 77.2–100%; false alarms 0.84–1.07% |
| Independent hard urgent lines, held-out set v4 (EN/ES/Haitian Creole): **139 / 162**; suicide risk **44 / 54** (before the fix: 118 and 25) | — | 79.6–90.3%; 69.2–89.6% (before: 33.7–59.4%) |

**Read this before relying on the red flag.** The sponsor data has only 13 urgent lines. So three independent sets of 162 hard urgent lines were written by a separate agent, each committed before its first run. The flag question's distress definition now names suicide warning signs (goodbyes and "in case" messages, giving things away, counting or saving pills, means, a plan, being a burden). On the held-out set v4 it catches **44 of 54** suicide-risk lines, against **25 of 54** before. The cost is false alarms: on the 25,785 real validation lines they rise from **9 to 244** (about one call in five). Most are medical (207). 116 of those are a single repeated line, "I need to get my sugar down". It still misses about 1 in 5 indirect warning signs. So every distress flag tells the coach **"Use your suicide-risk screening protocol now."** The coach's protocol stays in charge, and this flag is a backup ([Safety stress sets](#safety-stress-sets)).

English and Spanish are measured separately, and Spanish holds up. The score-vs-supervisors r is 0.818 in Spanish and 0.771 in English. Urgent lines caught: 7/7 in Spanish and 6/6 in English. **Haitian Creole is tested only on the safety flag** (stress sets, 41/54 urgent lines caught on the latest set). The sponsor data has no Creole calls, so tags and scores are not validated in Creole. The calls are synthetic (Crane Lab). On real, expert-labelled counselling sessions (AnnoMI), Clef tags 75.0% of lines like the experts' majority (68.2–79.6%), where the experts themselves agree 83.9% of the time. It almost never recognises client *sustain* talk there (2 of 31).

### Reproduce in 3 commands

```sh
git clone https://github.com/zaycruz/coach-copilot && cd coach-copilot
unzip /path/to/Challenge5_Coaches_Copilot.zip -d data/   # sponsor data, not in the repo (data/README.md)
scripts/reproduce.sh                                     # recomputes and checks 264 numbers, ~2 s, needs uv
```

The script rebuilds each number from committed model outputs, with no keys or network. Only latency, the GLM bake-off and the 30-cue live run are live-only. [REPRODUCE.md](REPRODUCE.md) maps each number to its file and key, and shows how to re-query the models. Live engine: https://coach.raava.dev · API: [CONTRACT.md](CONTRACT.md)

---

**Below:** [Architecture](#architecture) · [Clef payload](#clef-payload) · [Evaluation](#evaluation) · [Run](#run) · [Patient 360 endpoints](#patient-360-endpoints)

The engine follows a coaching call live or recorded, spoken or typed. It tags every line with a motivational-interviewing (MI) code and scores the coach. It raises a red banner for anything a person must act on today, and it gives the coach a cue only when one is needed.

## Architecture

```
speech (Safari = Apple speech engine) ─┐
recorded / typed lines ────────────────┼─> Worker (coach.raava.dev) ──> AI Gateway "coach-copilot" ──> Workers AI
                                       │     /api/turn   every line        Clef      (~0.3–0.5 s Clef call)
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

| Validation, 1,085 calls, 56,001 answered lines | Accuracy |
|---|---|
| Coach move (12-way) | **93.1%** (EN 93.9%, ES 91.6%) |
| Participant talk (3-way) | **90.9%** (EN 91.2%, ES 90.4%) |

95% CIs, bootstrapped over calls: move 92.8–93.4%, talk 90.5–91.3%. Always answering the most common label would score 20.3% (move) and 47.3% (talk). 37 of 56,038 lines got no answer (upstream errors during the batch run) and are left out. Counted as wrong, they give 93.0% and 90.8%. Cohen's kappa is **0.921** for coach move (EN 0.930, ES 0.904) and **0.848** for participant talk (EN 0.854, ES 0.839). These are model-vs-human kappas. The synthetic calls carry one human label per line, so the data has no human-vs-human line kappa to set them against (only call-level scores come from two supervisors); the closest human-human reference is AnnoMI's 10-expert subset below, where Clef's kappa is lower. The weakest labels are simple reflection (recall 73.0%; misses go to complex reflection, emphasize autonomy and affirm) and sustain talk (recall 63.4%). Per-label precision, recall, F1 and the full confusion matrices are in `runs/metrics.json` (`tags.validation.{move,talk}.labels`). Clef call time per line in the batch run (`evaluate.py` on `/judge`): p50 ~0.3–0.5 s, p95 ~1.05 s. A `/api/turn` round trip from a browser takes longer, because it adds the network and, on participant lines, waits for the parallel flag call. In `runs/e2e-live-20261017T170443Z.json`, measured from a Cloudflare sandbox, the round-trip p50 is about 310 ms for coach lines and 800 ms for participant lines (time inside the Worker: 278 and 688 ms). From a laptop client (`runs/e2e-live-20261017T171736Z.json`) it is about 680 ms and 720 ms (Worker 331 and 478 ms). The network path accounts for most of the gap. These numbers come from the move and talk questions as they ship. They were not rerun after `scope` joined the coach request, because `scope` is a separate question in the same call.

### Coach score vs supervisors (`scripts/agreement.py`, `GET /api/c5/agreement`)

There are 477 validation calls that both supervisors rated `overall_1_to_5`. The app score is the MITI-style score (`scripts/score.py`, each part rounded, then summed) over Clef's tags. The demo page rounds the sum instead, so it can differ by up to 2 points. CIs are a bootstrap over calls (2,000 resamples).

| Pair | Pearson r (95% CI) | Spearman ρ (95% CI) |
|---|---|---|
| **App score vs mean of supervisors A and B** | **0.789** (0.754–0.820) | **0.797** (0.758–0.830) |
| App score vs supervisor A | 0.751 (0.708–0.788) | 0.757 (0.711–0.794) |
| App score vs supervisor B | 0.701 (0.657–0.739) | 0.706 (0.655–0.748) |
| Supervisor A vs supervisor B | 0.692 (0.640–0.736) | 0.673 (0.614–0.724) |
| Score from human line labels vs mean of A and B | 0.816 (0.783–0.843) | 0.821 (0.786–0.850) |

**Like-for-like.** The mean of A and B (0.789) is smoother than any single rater, so it is not compared with A vs B. Instead, each supervisor is compared with the other supervisor:
- With supervisor A, the app agrees more closely than supervisor B does: r(app, A) − r(B, A) = +0.059 (paired bootstrap 95% CI 0.013–0.106).
- With supervisor B, the two are level: r(app, B) − r(A, B) = +0.009 (−0.032 to 0.056).

Cross-fitted ICC(2,1) gives the same picture. The app score is mapped to 1–5 by a line fitted on the other 4 of 5 folds, so no call's own ratings shape its mapped score. The ICC is 0.705 (0.658–0.743) with A and 0.656 (0.608–0.700) with B, against 0.666 (0.613–0.712) between A and B.

| By call language | Calls | App vs mean of A and B | App vs A | App vs B | A vs B |
|---|---|---|---|---|---|
| English | 312 | 0.771 (0.723–0.815) | 0.724 (0.670–0.775) | 0.680 (0.620–0.735) | 0.657 (0.581–0.720) |
| Spanish | 165 | 0.818 (0.772–0.858) | 0.797 (0.743–0.846) | 0.733 (0.673–0.785) | 0.746 (0.679–0.806) |

Supervisor A vs B: ICC(2,1) 0.666 (0.613–0.712), mean absolute difference 0.61 points on the 1–5 scale. Mapped to 1–5 by a least-squares line, the app score is off by 0.55 points on average from supervisor A and 0.60 from B. The score formula itself, run on perfect human labels, reaches r = 0.816, and the app's score correlates r = 0.955 with that. The endpoint returns the per-call table.

### Flags (`evaluate.py --flags-only`, `scripts/flag_eval.py`)

The participant flag is judged in its own line-only request. Its `none` examples come from dev false alarms only.

| | Result | Wilson 95% CI |
|---|---|---|
| `needs_human` lines caught | **13/13** (10 medical, 2 social, 1 distress) | 77.2–100% |
| False alarms, lines | **244 / 25,785** (22 distinct texts; 9 before the suicide-risk fix) | 0.84–1.07% |
| False alarms, calls | 228 / 1,085 (9 before the fix) | 18.7–23.5% |
| Coach conduct vs supervisor `safety_issue_handled` | 8/8 calls | 67.6–100% |
| English: caught / false-alarm lines / calls | 6/6 · 166 / 16,696 · 154 / 704 | 61.0–100% · 0.85–1.16% · 19.0–25.0% |
| Spanish: caught / false-alarm lines / calls | 7/7 · 78 / 9,089 · 74 / 381 | 64.6–100% · 0.69–1.07% · 15.8–23.7% |
| dev: caught / false alarms | 7/7, 53 / 6,854 lines | |

### Safety stress sets

Each line goes alone through the live `/api/turn` (`scripts/stress_flags.py`). An urgent line counts as caught when any red flag fires (medical, distress or social). Every set was committed before its first run.

| Set | Written by | Urgent caught | Distress (suicide risk) | Hard negatives flagged |
|---|---|---|---|---|
| v1, `stress-set.json`, EN/ES, 40 + 20 | us, before rules | 39/40 (87.1–99.6%) | 5/6 | 1/20 |
| v2, `stress-set-v2.json`, EN/ES/HT, 162 + 72 | separate agent from a brief, run with rules v1 | 132/162 (74.8–86.7%) | 31/54 (44.2–69.7%) | 31/72 (32.3–54.6%) |
| v3, `stress-set-v3.json`, EN/ES/HT, 162 + 72 | separate agent from a brief, run with current rules, previous definition | 131/162 (74.1–86.2%) | 28/54 (38.9–64.6%) | 21/72 (19.9–40.5%) |
| v4, `stress-set-v4.json`, EN/ES/HT, 162 + 72, previous definition (Worker ec884907) | separate agent from a brief, committed (e86bb99) before its first run, after the new definition was frozen | 118/162 (65.5–79.1%) | 25/54 (33.7–59.4%) | 20/72 |
| **v4, current definition** | same set, held out | **139/162 (79.6–90.3%)** | **44/54 (69.2–89.6%)** | **37/72** |

**Order of events.** The new definition (flag-question hash `f9319e93`) was written from v1–v3 misses and standard suicide warning-sign lists, and checked on v1–v3 only. There it gave distress 6/6, 52/54 and 44/54 (offline, `scripts/flag_revalidate.py`). No v4 line had been seen at that point. Then:
- v4 was committed (e86bb99, 18:24:20Z).
- A Worker preview with `f9319e93` was uploaded at 18:27:55Z (Cloudflare version 5977ff19, message "distress warning signs f9319e93").
- Only then was v4 run, twice and at the same time (18:30Z): on production with the old definition (118/162, distress 25/54) and on that preview with the new one (139/162, 44/54).
- Dev and validation were run after that, with the same frozen definition.

The definition did not change after v4 was run: the deployed Worker and `main` carry the same `f9319e93`, as the committed `runs/revalidate-*.log` files also show. The git history shows the definition committed (dc39d31) after v4 (e86bb99). That is because the change was committed together with its results, not because it was edited after v4. One more edit, adding validation-style false alarms to the `none` examples, was drafted after v4 and validation had been seen. It was reverted before any model call and was never committed or deployed. v4 and validation have now been seen, so any further change to the flag must be measured on a fresh set (Future work).

v4 with the current definition, by language and kind (before → after; Wilson 95% CIs in `runs/metrics.json` `stressV4`, `stressV4Before`):

| | Medical | Social | Distress |
|---|---|---|---|
| English | 18/18 | 17/18 | 10 → 15/18 |
| Spanish | 16/18 | 17/18 | 10 → 15/18 |
| Haitian Creole | 12/18 | 13 → 15/18 | 5 → 14/18 |

**What this says.**
- The flag now catches most indirect suicide warning signs in all three languages. It still misses about 1 in 5. Every distress flag carries `action: "Use your suicide-risk screening protocol now."`.
- The price is false alarms on real calls: 244 / 25,785 lines (0.84–1.07%), in 228 of 1,085 calls, against 9 before. 207 are medical and 36 distress. One repeated line, "I need to get my sugar down", accounts for 116. Lines like "I don't want to end up like my dad" now read as distress. We could not tune these away without touching the test split.
- Medical recall in Haitian Creole is the weakest cell (12/18 on v4).
- The independent sets' hard negatives are much harder than real calls (resolved ER visits, someone else's fever, dark humour): about half of v4's raise a flag.

**Model-free red-flag screen** (`src/redflags.json`, same patterns in `scripts/redflags.py`):
- What it is: patterns for explicit ideation, suicide warning signs, breathing or chest pain and bleeding, in English, Spanish and Haitian Creole. They run over lowercased, accent-stripped text before the model's answer is used.
- What it does: a hit sets that red flag and adds `rule` to the response. It never clears a flag.
- How it was built: tuned on dev and on v1 only. The warning-sign rules were added after v2's run, so v2 is not held out for them (distress 31/54 → 42/54 offline). v3 was written after them, and on v3 the rules caught 15 lines, 6 of them hard negatives.
- Real calls: on validation the rules add **no** false alarms and catch nothing the model missed (checked with both definitions).

**Re-validation of the distress definition (done, `scripts/flag_revalidate.py`):** the flag request was re-run on every validation and dev participant line (3,195 + 1,500 unique texts). It was throttled to 3–4 requests in flight, with an automatic stop on Workers AI rate limits and a pause during judges-panel evidence windows. 13/13 urgent lines are still caught.

### Coaching cues

Cues are written by GLM only at trigger moments that the page computes from the tags (`scripts/triggers.py`, the page's `checkTriggers` line for line).

**On the 1,085 held-out calls** (offline, from the committed tags and flags; `cues` in `runs/metrics.json`):
- **Rate:** 4.8 cues per call (median 5, 90th percentile 7), or one per 6 coach lines.
- **Trigger precision vs the human labels: 93.8% (93.1–94.4%).** Of the coaching moments where the app asks for a cue, 94% also meet the trigger under the human tags at that line. Recall is 98.6%.
- **By kind:** 78% are "change talk not reflected or affirmed" (precision 92.8%). Confront is the weakest at 87.2%.
- **Not a quality score.** The number of coaching cues per call correlates **+0.28** (Spearman, 95% CI 0.196–0.356) with the supervisors' mean score, so better-rated calls get *more* cues. The likely reason is that the dominant trigger fires on change talk, which better calls draw out more often. The trigger mix has not been validated by an MI supervisor.

**Live, 30 trigger moments** (6 per kind, seed 7; `scripts/cue_eval.py`, `runs/cue-eval-20261017T173558Z.json`):
- 30/30 answered, with no rate limits. GLM time in the Worker: **p50 2.4 s, p95 5.9 s** (round trip 3.0 s and 6.5 s).
- 30/30 `say` lines in the call's language, 30/30 with no dosing or medication instruction, and 30/30 cues of at most 45 words.
- Cues are asynchronous, so the tag and the banner never wait for them. Repeats are served from the KV cache.

**Behaviour change is not measured.** That needs coaches using it. The proposed pilot alternates cue-on and cue-off calls per coach, and the outcome is the reflection-to-question ratio and the reflection of change talk in the next 3 coach lines, as rated by supervisors.

### Second layer (GLM-5.3-flash)

`scripts/glm_bench.py` compared models on 60 dev lines where Clef's confidence was < 0.3 (Clef alone: 77%):

| Model | Review accuracy | Answered | Review p50 | Cue latency |
|---|---|---|---|---|
| GLM-5.3 | 62–74% | 37–58 / 60 | 3.7–4.2 s | 3–90 s, cold calls up to 48 s |
| **GLM-5.3-flash** | 78–84% | 27–32 / 60 | 2.2–3.1 s | 2–8 s |
| GLM-4.7-flash | 81–86% | 47–51 / 60 | 16–18 s | 7–22 s |

`scripts/review_eval.py` then ran 600 random validation lines with confidence < 0.3 through the live `/api/review`:
- Clef alone: 76.0%. Clef + review: 76.7% (fixed 8, broke 4).
- The review was right on 89.2% of the lines it answered, against Clef's 83.1% on those same lines.
- It answered only 65 of 600. The rest hit the Workers AI per-minute limit (error 3021).

Cues are where GLM earns its place. Review stays an optional background second opinion. Cue language is set by the Worker (`Call language: English|Spanish` from the participant's words); left to infer it, the models sometimes answered in Spanish or French.

### AnnoMI expert benchmark

`scripts/annomi_bench.py`: the 428 lines labelled by 10 experts (7 real counselling sessions), shipped payload unchanged, 2 previous lines of context. Clef's 12 moves are collapsed to AnnoMI's 4: questions and *ask permission* → question; simple/complex reflection → reflection; give information and persuade → information or advice; the rest → other.

| Lines | Clef | Experts agree with the majority label |
|---|---|---|
| counsellor (216) | 75.0% | 87.5% |
| client talk (212) | 75.0% | 80.2% |
| **all 428** | **75.0%** | **83.9%** |

Where all 10 experts agreed, Clef gets 91.1% (counsellor, 112 lines) and 93.1% (client, 58). Where experts split (≤ 70% agreement), Clef gets 50–58%. Most misses are client change or sustain talk read as neutral (37 lines). Cohen's kappa: counsellor **0.656**, client **0.439**. Client sustain talk is almost never recognised (recall 2 of 31). Reflections are under-called (recall 51.1%), and information or advice is under-called too (52.4%). Confusion matrices: `annomi.labels` in `runs/metrics.json`. Subtypes, scored where the coarse label matched: open/closed 73.2% (56; lines Clef tagged *ask permission* have no subtype and are left out), simple/complex 39.1% (23). The payload was tuned on synthetic cancer-coaching calls, not on AnnoMI.

## Run

```sh
npm install
npx wrangler deploy                                   # coach.raava.dev + workers.dev
scripts/reproduce.sh                                  # every number above, offline, checked (REPRODUCE.md)
uv run python -I scripts/e2e_live.py                  # live checks + drift check against runs/metrics.json
```

[REPRODUCE.md](REPRODUCE.md) has the data step, the table from README numbers to `runs/metrics.json` keys, and the commands that re-query the models.

The `/judge` endpoint is a raw Clef proxy for batch scripts (`Authorization: Bearer $CLEF_API_KEY` from `.env`).

**Rate limits.** Workers AI caps requests per minute per model (error 3021). The tag path (`/api/turn`) waits about 0.5 s, then about 1.5 s (jittered), before answering `429`, so a live tag is late rather than lost in a short burst. The async GLM layer backs off (about 2 s, then about 5 s, jittered) and retries twice. Then it answers `429` with `retry-after: 20`. Review and cue answers are cached in KV for 7 days, keyed by model and prompt, so replaying a call costs no quota (`cached: true` in the response). `GET /api/c5/demo` serves a pre-played safety call with its cues and makes no model calls. The UI keeps Clef's tag when a review gets no label. Batch scripts against the live Worker share this quota with live calls and with Patient 360, so do not run them during a demo.

## Patient 360 endpoints

`src/c5.ts` serves the suite endpoints that Patient 360 (coach-console) asks for in its `INTEGRATION.md`. They use the same shape as the console's adapter, so the console switched over with no change (`x-c5-source: native`):

- `GET /api/c5/calls[/:callId]`: the 27 supervisor-labelled demo calls, scored like `scripts/score.py`.
- `GET /api/c5/survivors[/:survivorId]`: one row per participant (`c5-P0540`), with status from their worst call.
- `GET /api/c5/demo`: held-out safety call CALL05423, pre-played through the live engine (`scripts/preplay.py`). It has every line's Clef tag and flag, the red medical banner at 05:12, 4 GLM cues, and score 74 (82 on the supervisor labels). 90.6% of its tags match the human labels. A console can open on it with no model calls.
- `GET /api/c5/agreement`: the supervisor-agreement figures above, with the per-call table (app score, human-label score, supervisor A, supervisor B) and the flag Wilson CIs.

The field-level contract, with real responses, is in [`CONTRACT.md`](CONTRACT.md) (`scripts/contract.py`).

The flag kind (medical, social…) on each `needs_human` line is Clef's answer to the line-only flag question, stored in `public/demo-calls.json`. The console's adapter guessed it from keywords; the answers are the same on all 9 lines.

## Future work

- **Fewer false alarms from the suicide-risk fix. This needs a fresh validation set.** Most of the 244 validation false alarms are plain goal and motivation lines. "I need to get my sugar down" alone is 116 of them, followed by "I don't want to end up like my dad" and similar. Adding such lines to the flag's `none` examples would likely cut them. But validation and v4 have now been seen, so any number from that change would be post hoc. It has to be measured on a new, untouched validation set and a new independent stress set, with a stated operating point (calls with a false alarm).
- **Cue usefulness.** An MI-trained supervisor rates a sample of cues (useful / ignorable / interrupting), reported per cue type, with a quieter cue budget.
- **Haitian Creole.** A small labelled Creole call set, to validate tags and scores, not only the safety flag.
- **Clinician-written stress sets.** The independent sets were written by a separate LLM agent, not by clinicians.

## Sources

- AnnoMI: Wu Z, et al. *Creation, Analysis and Evaluation of AnnoMI, a Dataset of Expert-Annotated Counselling Dialogues.* Future Internet, 2023.
- Text coaching corpus: Gupta I, et al. *Human-Human Health Coaching via Text Messages: Corpus, Annotation, and Analysis.* SIGDIAL 2020.
- Synthetic data: Crane Lab, Sylvester Comprehensive Cancer Center.
