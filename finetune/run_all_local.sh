#!/bin/zsh
# seed 0 for all three conditions first, then seed 1
cd "$(dirname "$0")/.."
for s in 0 1; do
  .venv/bin/python -u finetune/eval_local.py base $s || exit 1
  VACCINE_ADAPTER=finetune/adapter_raw .venv/bin/python -u finetune/eval_local.py ft $s || exit 1
done
