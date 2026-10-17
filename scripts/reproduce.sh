#!/usr/bin/env bash
# Rebuild every README headline number offline (Tier A in REPRODUCE.md) and check it.
# Needs: uv, and the sponsor data unzipped under data/ (see data/README.md). No network after `uv sync`, no keys.
# Usage: scripts/reproduce.sh [data_dir]   (default data/Challenge5_Coaches_Copilot)
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${1:-data/Challenge5_Coaches_Copilot}"

echo "== 1/3 sponsor data hashes (data/SHA256SUMS)"
if command -v sha256sum >/dev/null; then (cd data && sha256sum -c SHA256SUMS); else (cd data && shasum -a 256 -c SHA256SUMS); fi
[ "$DATA" = data/Challenge5_Coaches_Copilot ] || echo "note: hashes checked under data/, metrics read from $DATA"

echo "== 2/3 metrics from runs/predictions + sponsor CSVs -> runs/metrics.json"
uv sync --frozen --quiet
start=$(date +%s)
uv run --frozen python -I scripts/metrics.py "$DATA" >/dev/null
echo "metrics.py took $(( $(date +%s) - start ))s"

echo "== 3/3 compare with README numbers (scripts/expected.json)"
uv run --frozen python -I scripts/check_metrics.py
