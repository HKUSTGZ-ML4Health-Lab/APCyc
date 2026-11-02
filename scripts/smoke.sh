#!/usr/bin/env bash
# Run from an environment installed according to README.md. No training is invoked.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
device="${1:-cpu}"
output="${2:-outputs/smoke}"
require_bitwise="${APCYC_REQUIRE_BITWISE:-auto}"
case "$require_bitwise" in
  auto|0|1) ;;
  *) echo 'APCYC_REQUIRE_BITWISE must be auto, 0, or 1.' >&2; exit 2 ;;
esac
: "${APCYC_INPUT:?Set APCYC_INPUT to an explicit input manifest; use scripts/make_example.py for a synthetic fixture}"
input="$APCYC_INPUT"
bundle_path="${APCYC_BUNDLE:-${APCYC_PRIVATE_ROOT:+$APCYC_PRIVATE_ROOT/weights/apcyc.pt}}"
if [ -z "$bundle_path" ]; then
  echo 'Set APCYC_BUNDLE to a trained generator bundle.' >&2
  exit 2
fi
config_path="${APCYC_CONFIG:-configs/inference/default.yaml}"
comparison_bundle="${APCYC_COMPARISON_BUNDLE:-}"
if [ -e "$output" ]; then
  echo 'Choose a new output directory.' >&2
  exit 1
fi
mkdir -p "$output"
roles=(default repeat)
if [ -n "$comparison_bundle" ]; then roles+=(comparison); fi
for role in "${roles[@]}"; do
  selected_bundle="$bundle_path"
  selected_config="$config_path"
  if [ "$role" = comparison ]; then
    selected_bundle="$comparison_bundle"
    selected_config="${APCYC_COMPARISON_CONFIG:-configs/inference/default.yaml}"
  fi
  python -m apcyc.sample --bundle "$selected_bundle" \
    --input "$input" --config "$selected_config" \
    --out-dir "$output/$role" --device "$device" --seed 2021
  python scripts/inspect_predictions.py "$output/$role" --output "$output/$role/inspection.json"
done
python - "$output" "$device" "$require_bitwise" <<'PY'
from pathlib import Path
import json,sys
root=Path(sys.argv[1])
a,b=root/'default',root/'repeat'
rows=[json.loads(s) for s in (a/'predictions.jsonl').read_text().splitlines()]
checks={f:(a/f).read_bytes()==(b/f).read_bytes() for f in ['predictions.jsonl']+[r['pdb'] for r in rows]}
(root/'repeat_check.json').write_text(json.dumps(checks,indent=2)+'\n')
requested=sys.argv[3]
require_bitwise=(not sys.argv[2].startswith('cuda')) if requested=='auto' else requested=='1'
report={'device':sys.argv[2], 'bitwise_equal':all(checks.values()),
        'require_bitwise':require_bitwise, 'checks':checks}
(root/'repeat_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
if require_bitwise and not all(checks.values()):
    raise SystemExit('Repeated outputs differ and bitwise equality was required.')
PY
