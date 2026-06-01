# Examples

The commands below mirror the root README while keeping all paths explicit.

## Pocket Extraction

```bash
python -m api.detect_pocket \
  --pdb assets/1ssc_A_B.pdb \
  --target_chains A \
  --ligand_chains B \
  --out assets/1ssc_A_pocket.json
```

## Codesign

```bash
CKPT=checkpoints/codesign.ckpt bash bash/sample_codesign.sh
```

## Fixed-Sequence Structure Prediction

```bash
CKPT=checkpoints/fixseq.ckpt bash bash/sample_struct_pred.sh
```
