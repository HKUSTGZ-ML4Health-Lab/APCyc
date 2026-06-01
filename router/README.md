Cyclization Router (Topology Generator / Soft Router)
====================================================

This module trains a hierarchy-aware router to predict cyclization topology
T = (c, i, j) with p(T | G_b) = p(c | G_b) * p(i, j | G_b, c).

Inputs
------
- JSONL file with fields:
  - sample_id (string)
  - peptide_length (int)
  - c_star (string, e.g. "HT", "SS", "ISO")
  - gt_pair (list [i, j])
  - type_soft (list of floats, same order as --type_list)
  - pair_soft (list of [i, j, w])
- PDB directory with files named {sample_id}.pdb
  - Receptor chain is "R" by default (set --chain_id to override)

Graph Construction (aligned with APCyc)
-----------------------------------------
- Residue-level nodes with multi-atom coordinates (X: [N, 14, 3] by default).
- Residue types use APCyc VOCAB; atom channels match APCyc ordering.
- Edges: full intra-graph connections (same as dyMEAN using variadic_meshgrid).
- Use --backbone_only to restrict to 4 backbone atoms.

Valid Pair Rules
----------------
Default rule (type-dependent):
- HT: (0, N-1)
- SS: all i < j (min_sep applies)
- ISO: all i < j (min_sep applies)

Use --pair_rule to switch:
- default | all_pairs | end_to_end | nterm

Cross Attention
---------------
Enabled by default. Use --no_cross_attention to disable.

Type Conditioning
-----------------
--type_condition add|film
add: h = h + type_embedding(c)
film: h = h * (1 + gamma) + beta

Training
--------
Config-driven (recommended for multi-run):
  python train_router.py \
    --config configs/router/paradigm/stage123.yaml \
    --train_jsonl /path/to/cpcore_train.jsonl \
    --pdb_dir /path/to/CPCore_pdb

Scheduler (optional):
  --use_scheduler --scheduler_class ReduceLROnPlateau

Stage-wise (recommended):
  python train_router.py \
    --train_jsonl /path/to/cpcore_train.jsonl \
    --pdb_dir /path/to/CPCore_pdb \
    --train_mode stage \
    --type_list HT,SS,ISO \
    --lambda1 1.0 --lambda2 1.0

Backbone-only (4 atoms):
  --backbone_only

Use predefined splits (train/valid txt with PDB ids):
  python train_router.py \
    --train_jsonl /path/to/cpcore_train.jsonl \
    --pdb_dir /path/to/CPCore_pdb \
    --train_ids /path/to/CPCore_train.txt \
    --valid_ids /path/to/CPCore_valid.txt

Joint:
  python train_router.py \
    --train_jsonl /path/to/cpcore_train.jsonl \
    --pdb_dir /path/to/CPCore_pdb \
    --train_mode joint \
    --joint_type_condition gt

Finetune (soft targets only):
  python train_router.py \
    --train_jsonl /path/to/cpcore_train.jsonl \
    --pdb_dir /path/to/CPCore_pdb \
    --train_mode finetune \
    --use_type_soft --use_pair_soft

Optional Stage 3:
  --enable_stage3 --stage3_type_condition mix --stage3_pred_prob 0.5

Pretrained / Freeze
-------------------
--pretrained_mode none|full|partial
--pretrained_ckpt /path/to/apcyc.ckpt
--freeze_mode none|all|partial
--freeze_layers 2

TensorBoard
-----------
--log_dir ./router_logs
TensorBoard tags:
  stage1_type/*, stage2_site/*, stage3_joint/*, joint/*

Memory-Mapped Cache (optional)
------------------------------
Build cache (separate step):
  python router/build_router_mmap.py \
    --train_jsonl /path/to/cpcore_train.jsonl \
    --pdb_dir /path/to/CPCore_pdb \
    --mmap_dir /path/to/router_mmap

Use cache:
  --use_mmap --mmap_dir /path/to/router_mmap

Outputs
-------
Checkpoints saved under --output_dir:
  stage1_type_best.pt, stage2_site_best.pt, stage3_joint_best.pt, joint_best.pt
