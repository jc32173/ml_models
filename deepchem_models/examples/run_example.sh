#! /bin/bash

# Remove any files from previous run:

rm -fr train_val_test_split.csv predictions.csv \
       GCNN_info_all_models.csv GCNN_info_refit_models.csv \
       GraphConvModel_info_av_performance.csv \
       resample_0_cv_refit_model_1 resample_1_cv_refit_model_0

# Run example using parameters in run_params.json file and
# with random seed set to 0 to make results reproducible:

python ../scripts/run_deepchem_models.py run_params.json 0
