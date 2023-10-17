#! /bin/bash

# Calculate 

python ../preds_script_argparse.py \
    -i test_inchis.inchi.gz \
    -l 0 100 \
    -c 20 \
    -o all_preds \
    -e invalid_inchis.inchi.gz \
    --lilly_rules \
    --lilly_rules_script /users/xpb20111/repos/PP/src/Lilly-Medchem-Rules/Lilly_Medchem_Rules.rb \
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
    --calc_oe_logp \
    --calc_pfi \
    --calc_mpo \
    --models "pIC50_pred" "pIC50.pk" \
             "logD_pred" "logD.pk"
