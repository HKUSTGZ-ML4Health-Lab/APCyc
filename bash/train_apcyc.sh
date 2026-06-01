#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/apcyc/apcyc_ae.yaml}"
shift || true

GPU="${GPU:-0}" bash scripts/train.sh "${CONFIG}" "$@"
