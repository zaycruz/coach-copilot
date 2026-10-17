# Reproduce the C5 numbers

There are two tiers.
- **Tier A** rebuilds every README number offline from the committed model outputs. It is exact, needs no network and no keys, and runs in seconds on a laptop CPU.
- **Tier B** re-queries the models to regenerate those outputs. It needs your own Worker and a Workers AI quota. GLM is stochastic, so its numbers move between runs.

## Tier A: one command

**Prerequisites:** git, [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and the pinned `pandas==2.2.3` and `numpy==2.2.6` from `uv.lock`), and `sha256sum` or `shasum`.

**Data:** the sponsor pack `Challenge5_Coaches_Copilot.zip`, which is not in the repo. `data/README.md` gives the zip's sha256, the four files used, their hashes, and the public AnnoMI source.

```sh
git clone https://github.com/zaycruz/coach-copilot && cd coach-copilot
unzip /path/to/Challenge5_Coaches_Copilot.zip -d data/
scripts/reproduce.sh
```

`reproduce.sh` runs three steps:
1. It checks the four data files against `data/SHA256SUMS` and stops on any mismatch.
2. It runs `scripts/metrics.py`, which joins `runs/predictions/*.jsonl` (ids and model answers only, no text) with the sponsor labels and writes `runs/metrics.json` and `runs/agreement-per-call.json`.
3. It runs `scripts/check_metrics.py`, which compares 178 values with `scripts/expected.json`, rounded as the README shows them, and exits 1 on any mismatch.

**Runtime:** about 2 s for the metrics on an M-series laptop, plus the one-time `uv sync`.

**Determinism:** the bootstrap CIs use 2,000 resamples with `numpy.random.default_rng(7)`, and the dev/validation split is `sha1(participant_id) % 5 == 0` → dev. There is no other randomness.

### README number → `runs/metrics.json` key

The full list (178 keys) is `scripts/expected.json`. The headline numbers:

| README number | Key | Expected |
|---|---|---|
| Coach move accuracy (EN, ES) | `tags.validation.move.accuracy` (`.byLanguage.English`, `.Spanish`) | 0.931 (0.939, 0.916) |
| Coach move tags: 95% CI (bootstrap over calls); always-most-common-label baseline | `tags.validation.move.accuracyCi95`; `tags.validation.move.majorityBaseline.accuracy` | 0.928–0.934; 0.203 |
| Urgent base rate | `flags.validation.urgentPrevalence.rate` | 0.0005 |
| AnnoMI accuracy, 95% CI (bootstrap over the 7 sessions) | `annomi.accuracyCi95` | 0.682–0.796 |
| Participant talk accuracy (EN, ES) | `tags.validation.talk.accuracy` (`.byLanguage.*`) | 0.909 (0.912, 0.904) |
| Lines without an answer; accuracy with them counted wrong | `tags.validation.noAnswer`; `tags.validation.{move,talk}.accuracyCountingNoAnswerWrong` | 37; 0.930, 0.908 |
| Calibration: share of lines at confidence ≥ 0.3, accuracy there, accuracy below (dev) | `tags.dev.calibration.*` | 0.77, 0.983, 0.76 |
| App score vs mean of supervisors: Pearson (CI), Spearman | `agreement.correlations.2.pearson` (`.pearsonCi95`), `.spearman` | 0.789 (0.754–0.820), 0.797 |
| Like-for-like: r(app, A) − r(B, A); r(app, B) − r(A, B) | `agreement.likeForLike.*.diff` (`.ci95`) | +0.059 (0.013–0.106); +0.009 (−0.032 to 0.056) |
| Cross-fitted ICC(2,1): app vs A, app vs B | `agreement.crossFittedIcc.{appVsA,appVsB}.icc21` | 0.705, 0.656 |
| By language: app vs mean, English and Spanish | `agreement.byLanguage.{English,Spanish}.appVsMean.pearson` | 0.771, 0.818 |
| Flags by language: caught, English and Spanish | `flags.validation.byLanguage.{English,Spanish}.caught` | 6/6, 7/7 |
| App vs supervisor A, app vs B, A vs B | `agreement.correlations.{0,1,3}.pearson` | 0.751, 0.701, 0.692 |
| Human-label score vs mean; app vs human-label score | `agreement.correlations.{4,5}.pearson` | 0.816, 0.955 |
| Supervisor A vs B, ICC(2,1) (CI) | `agreement.supervisorIcc.icc21` (`.ci95`) | 0.666 (0.613–0.712) |
| MAE: app vs A, app vs B, A vs B | `agreement.mae.{maeVsA,maeVsB,maeAvsB}` | 0.553, 0.597, 0.61 |
| `needs_human` lines caught (Wilson CI) | `flags.validation.caught` | 13/13 (0.772–1) |
| False-alarm lines (CI), distinct texts, dropout | `flags.validation.falseAlarmLines`, `.falseAlarmTexts`, `.falseAlarmDropout` | 9/25,785 (0.0002–0.0007), 5, 0 |
| Calls with a false alarm | `flags.validation.falseAlarmCalls` | 9/1,085 (0.004–0.016) |
| Coach conduct vs supervisors | `flags.coachConduct` | 8/8 (0.676–1) |
| Dev: lines caught, false alarms | `flags.dev.caught`, `flags.dev.falseAlarmLines` | 7/7, 1/6,854 |
| AnnoMI: Clef vs expert agreement, all / counsellor / client | `annomi.accuracy`, `annomi.expertAgreementPct` (and per speaker) | 0.750 vs 83.9; 0.750 vs 87.5; 0.750 vs 80.2 |
| AnnoMI, lines where all 10 experts agreed | `annomi.{counsellor,client}.byAgreement.100.accuracy` | 0.911, 0.931 |
| AnnoMI subtypes | `annomi.questionSubtype`, `annomi.reflectionSubtype` | 0.732 (56), 0.391 (23) |
| GLM review on 600 low-confidence lines | `review.{clefAlone,clefPlusReview,fixed,broke,answered}` | 0.760 → 0.767, fixed 8, broke 4, answered 65 |

`src/agreement.json`, which `GET /api/c5/agreement` serves, is written by `scripts/agreement.py` from the same functions in `metrics.py`.

### Not in Tier A

These numbers are live-only, so they are not recomputed offline:
- **The GLM model bake-off** (`scripts/glm_bench.py`, README "Second layer" table). It is stochastic and rate-limited, and the README reports it as ranges over repeated runs.
- **Latency.** It depends on the network and on load. `scripts/e2e_live.py` measures it.
- **The pre-played demo call** (`scripts/preplay.py` → `src/demo-safety-call.json`). The GLM cues differ on every run.
- **`scripts/spike.py`.** It was the first payload experiment and is kept for history. No README number comes from it.

## Tier B: regenerate the model outputs

1. Deploy your own Worker: `npm install`, `npx wrangler deploy` (your Cloudflare account, with Workers AI enabled), then `npx wrangler secret put API_KEY`.
2. Point the scripts at it: `export COACH_COPILOT_URL=https://<your-worker>.workers.dev CLEF_API_KEY=<the same key>`.
3. Re-query. These write the full local outputs, which include text and are gitignored, under `runs/`:
   ```sh
   D=data/Challenge5_Coaches_Copilot
   uv run python -I scripts/evaluate.py $D dev 120 @cf/cloudflare/clef 2            # tags, dev: 120 calls drawn with random_state=7
   uv run python -I scripts/evaluate.py $D validation 0 @cf/cloudflare/clef 2       # tags, validation
   uv run python -I scripts/evaluate.py $D dev 0 @cf/cloudflare/clef 2 --flags-only   # flags, all 278 dev calls
   uv run python -I scripts/evaluate.py $D validation 0 @cf/cloudflare/clef 2 --flags-only
   uv run python -I scripts/flag_eval.py $D > runs/flag-eval-v2.log                  # coach conduct
   uv run python -I scripts/annomi_bench.py $D
   uv run python -I scripts/review_eval.py $D 600                                   # GLM; rate-limited
   ```
4. Run `uv run python -I scripts/export_predictions.py` to write the text-free `runs/predictions/`, then `scripts/reproduce.sh`.

Clef is near-deterministic, so tags and flags should land within a few lines of the committed outputs. The review result depends on how many GLM calls get past the per-minute quota. The committed run answered 65 of 600.

The Workers AI quota is shared by every caller of a Worker. Do not run Tier B against coach.raava.dev.

## Live E2E

```sh
uv run python -I scripts/e2e_live.py [baseUrl]    # default https://coach.raava.dev
```

The script needs no key and makes no GLM calls, only about 12 Clef calls. It checks:
- health
- the red flag on English and Spanish fever-on-chemo lines
- no flag on everyday lines
- `missed_concern` on a coach line that moves past a flag
- the 400 and 404 errors
- the shape of `/api/c5/calls`, `/api/c5/survivors` and `/api/c5/demo`
- a **drift check**: every number served by `/api/c5/agreement` must equal the repo's `runs/metrics.json` and `runs/agreement-per-call.json`

It also records latency, both the Worker's own Clef time and the client round trip.

The artifact is `runs/e2e-live-<UTC stamp>.json`, and the script exits 1 on any failed check.

## Proof runs

`runs/reproduce-<UTC stamp>.md` records a fresh-clone run of Tier A: where it ran, the commands, the tool versions, the check table and the result. The 2026-10-17 run used Cloudflare Containers, via the throwaway Worker in `scripts/cloudflare-repro/`:

```sh
cd scripts/cloudflare-repro && npm install && npx wrangler deploy
openssl rand -hex 24 > .secret && npx wrangler secret put RUN_SECRET < .secret
curl -X POST -H "authorization: Bearer $(cat .secret)" --data-binary @Challenge5_Coaches_Copilot.zip \
  "https://c5-repro-proof.<account>.workers.dev/run?commit=<sha>&e2e=https://coach.raava.dev"
npx wrangler delete
```
