#! /bin/bash

# Calculate model predictions and descriptors to check that results have not changed.

python ../preds_script_argparse.py \
    -i test_inchis.inchi.gz \
    -l 0 100 \
    -c 20 \
    -o all_preds \
    -e invalid_inchis.inchi.gz \
    --lilly_rules \
    --lilly_rules_script ${lilly_rules_script} \
    --desc "molwt" \
           "n_aromatic_rings" \
           "n_heavy_atoms" \
           "murcko_scaffold" \
           "SAscore" \
           "max_fused_rings" \
    --hist "pIC50_pred" 0.1 \
           "MPO" 0.1 \
           "molwt" 5 \
           "n_heavy_atoms" 1 \
    --hist_by_substruct hydantoin_substructs.csv \
    --calc_oe_logp \
    --calc_pfi \
    --calc_mpo \
    --models "pIC50_pred" "pIC50.pk" \
             "logD_pred" "logD.pk"
