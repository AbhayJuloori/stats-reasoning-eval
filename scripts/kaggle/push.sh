#!/usr/bin/env bash
# Packs code + prompts + data into a private Kaggle dataset and pushes the GPU kernel.
set -euo pipefail
cd "$(dirname "$0")/../.."
USER_NAME=$(kaggle config view 2>/dev/null | awk -F': ' '/username/ {print $2}')
STAGE=$(mktemp -d)
mkdir -p "$STAGE/dataset" "$STAGE/kernel" "$STAGE/bundle/data/sheets" "$STAGE/bundle/data/raw/QRData/benchmark" "$STAGE/bundle/results/logs"
cp -R evalset prompts rubric "$STAGE/bundle/"
cp -L data/sheets/*.csv "$STAGE/bundle/data/sheets/"
cp data/raw/QRData/benchmark/QRData.json "$STAGE/bundle/data/raw/QRData/benchmark/"
find "$STAGE/bundle" -name __pycache__ -prune -exec rm -rf {} +
tar -czf "$STAGE/dataset/bundle.tar.gz" -C "$STAGE/bundle" .

cat > "$STAGE/dataset/dataset-metadata.json" <<JSON
{"title": "stats-reasoning-eval-bundle", "id": "$USER_NAME/stats-reasoning-eval-bundle", "licenses": [{"name": "other"}]}
JSON
if kaggle datasets status "$USER_NAME/stats-reasoning-eval-bundle" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE/dataset" -m "update bundle" -q
else
  kaggle datasets create -p "$STAGE/dataset" -q
fi

cp scripts/kaggle/kernel.py "$STAGE/kernel/kernel.py"
cat > "$STAGE/kernel/kernel-metadata.json" <<JSON
{"id": "$USER_NAME/stats-reasoning-eval-solve", "title": "stats-reasoning-eval-solve", "code_file": "kernel.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "machine_shape": "NvidiaTeslaT4", "dataset_sources": ["$USER_NAME/stats-reasoning-eval-bundle"]}
JSON
echo "waiting for the dataset to be ready"
until kaggle datasets status "$USER_NAME/stats-reasoning-eval-bundle" 2>/dev/null | grep -q ready; do sleep 10; done
kaggle kernels push -p "$STAGE/kernel"
