#! /bin/bash

# Run test by training models and comparing the results to a previous run:

python ../scripts/run_deepchem_models.py run_params.json 0 > /dev/null

python <<EOF
import pandas as pd

tolerance = 0.001

print('Results difference tolerance:', (tolerance))

df1 = pd.read_csv('GraphConvModel_info_av_performance_ref.csv', 
                  index_col=[0, 1])
df2 = pd.read_csv('GraphConvModel_info_av_performance.csv', 
                  index_col=[0, 1])

diff_below_tol = bool((df1 - df2).max().max() < tolerance)

print('Difference in results:')
print(df1 - df2)

print('\n================')
print('Test pass: {:>5}'.format(str(diff_below_tol)))
print('================\n')

EOF

mv GraphConvModel_info_av_performance.csv GraphConvModel_info_av_performance_last_run.csv
rm GCNN_info_all_models.csv GCNN_info_refit_models.csv train_val_test_split.csv 
