# Data

Nothing in this folder except this file and `SHA256SUMS` is committed. The sponsor data is not ours to publish.

## Sponsor data (required)

`Challenge5_Coaches_Copilot.zip`, the Emerge Hackathon Challenge 5 pack (Crane Lab, Sylvester Comprehensive Cancer Center), sha256 `68b6ba63994992e25a9f5bd77adb339c8464decfb9664a2e3b7647d06172747c`. It comes from the hackathon organisers, and access is through them. Unzip it here:

```sh
unzip Challenge5_Coaches_Copilot.zip -d data/      # -> data/Challenge5_Coaches_Copilot/...
shasum -a 256 -c data/SHA256SUMS                     # run from the repo root
```

The four files the scripts read:

| Path under `data/` | Rows (with header) | Used for |
|---|---|---|
| `Challenge5_Coaches_Copilot/synthetic_data/synthetic_call_transcripts.csv` | 288,274 | line labels: `coach_move`, `participant_talk`, `needs_human`, text |
| `Challenge5_Coaches_Copilot/synthetic_data/synthetic_calls.csv` | 5,571 | `labelled`, `language`, `participant_id`, which sets the dev/validation split (`sha1(participant_id) % 5 == 0` → dev) |
| `Challenge5_Coaches_Copilot/synthetic_data/synthetic_supervisor_ratings.csv` | 1,201 | two supervisors' `overall_1_to_5` and `safety_issue_handled` |
| `Challenge5_Coaches_Copilot/coaching_sessions_annomi.csv` | 9,701 | AnnoMI lines with `expert_raters` and `expert_agreement_pct` (the 428 ten-expert lines) |

`reproduce.sh` checks these hashes and stops on any mismatch.

## Public real data

- **AnnoMI** (Wu Z, et al., *Future Internet* 2023; public domain): https://github.com/uccollab/AnnoMI, commit `42936645ec3857a9c84ab296a36a3c34b779ef49` (2023-03-14). No access request is needed.
  - The public `AnnoMI-full.csv` has one row per annotator. It does not carry the sponsor file's `expert_raters` and `expert_agreement_pct` columns.
  - So the benchmark uses the sponsor copy above, which is AnnoMI with those two columns added.
