#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/tedimol_s2_10k
PROJECT="$BASE/00_共用第一阶段底座与训练工具"
SOURCE=/root/autodl-tmp/tedimol_chembi
PARENT="$SOURCE/runs/chembi_seed19991009/latest.pt"
RUN_ROOT="$PROJECT/runs_s2_10k_seed19991009"

mkdir -p "$PROJECT"
for f in "$SOURCE"/*.py; do
  name=$(basename "$f")
  case "$name" in
    comparison_runtime.py|method_registry.py|property_common.py|property_data.py|property_metrics.py|property_model_base.py|property_runtime.py|sample_compare.py|train_compare.py|sample_s2.py|train_s2.py) continue ;;
  esac
  ln -sfn "$f" "$PROJECT/$name"
done
ln -sfn "$SOURCE/improved_diffusion" "$PROJECT/improved_diffusion"
ln -sfn "$SOURCE/data" "$PROJECT/data"
ln -sfn "$SOURCE/models" "$PROJECT/models"

source /root/autodl-tmp/tedimol_env/bin/activate
cd "$PROJECT"
export PYTHONPATH="$PROJECT:$BASE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
exec > >(tee "logs/s2_launcher.log") 2>&1

python - <<'PY'
from pathlib import Path
import torch
from common import MoleculeTokenizer, open_cache
from property_runtime import _torch_load, validate_a0_checkpoint, property_protocol_receipt, DEFAULT_PROPERTY_DIR
parent = Path('/root/autodl-tmp/tedimol_chembi/runs/chembi_seed19991009/latest.pt')
assert parent.is_file(), parent
assert torch.cuda.is_available(), 'CUDA unavailable'
checkpoint = _torch_load(parent)
validate_a0_checkpoint(checkpoint, MoleculeTokenizer())
property_protocol_receipt(DEFAULT_PROPERTY_DIR, ('logP','SA','TPSA'))
rows, states, masks, meta = open_cache('train_ChemBI', 0)
assert len(rows) == 22169 and meta['complete']
print('S2_PREFLIGHT_PASS')
print('parent_step=', checkpoint['step'])
print('train_rows=', len(rows))
print('gpu=', torch.cuda.get_device_name(0))
PY

for MODE in a0c a1_prefix a2_rt_numeric a3_adaln_zero; do
  python -u "$BASE/train_s2.py" --mode "$MODE" --steps 10000 \
    --parent "$PARENT" --run "$RUN_ROOT/$MODE" \
    --acknowledge-new-protocol
done

for MODE in a0c a1_prefix a2_rt_numeric a3_adaln_zero; do
  for WEIGHTS in RAW EMA; do
    python -u "$BASE/sample_s2.py" --checkpoint "$RUN_ROOT/$MODE/latest.pt" \
      --weights "$WEIGHTS" --expected-steps 10000 \
      --output "$RUN_ROOT/$MODE/${MODE}_${WEIGHTS}_step10000.csv"
  done
done

python - <<'PY'
from pathlib import Path
import json, hashlib
root = Path('/root/autodl-tmp/tedimol_s2_10k/00_共用第一阶段底座与训练工具/runs_s2_10k_seed19991009')
items=[]
for p in sorted(root.rglob('*')):
    if p.is_file() and p.name != 's2_file_manifest.json':
        items.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(root/'s2_file_manifest.json').write_text(json.dumps({'protocol':'tedimol-s2-10k-20261012-v1','files':items},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('S2_RESULT_MANIFEST_PASS', len(items))
PY

STAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE="/root/autodl-tmp/TediMol_S2_10k_results_${STAMP}.tar.gz"
tar -czf "$PACKAGE" -C "$BASE" "00_共用第一阶段底座与训练工具/runs_s2_10k_seed19991009"
(
  cd /root/autodl-tmp
  sha256sum "$(basename "$PACKAGE")" > "${PACKAGE}.sha256"
  sha256sum -c "${PACKAGE}.sha256"
)
ls -lh "$PACKAGE" "${PACKAGE}.sha256"
echo 'S2_10K_FULL_PASS'
echo "RESULTS_PACKAGE=$PACKAGE"
