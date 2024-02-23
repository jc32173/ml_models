#! /bin/bash

# Run preds_script.py on test data and compare the resulting files to a 
# previous run:

## Update reference files:
#echo "Running preds_script.py..."
#bash test_calc_preds.sh
#mv all_preds_0-100.csv.gz all_preds_0-100_ref.csv.gz
#mv invalid_inchis.inchi.gz invalid_inchis_ref.inchi.gz
#mv mol_counts.csv mol_counts_ref.csv
#for prpty in "pIC50_pred" "MPO" "molwt" "n_heavy_atoms"
#do
#    for suffix in "" "_by_substructure"
#    do
#        mv hist_${prpty}${suffix}.csv hist_${prpty}${suffix}_ref.csv
#    done
#done

# Run test:

echo "Running preds_script.py..."
bash test_calc_preds.sh #&> /dev/null

# Check output files using diff:

echo "Running diff on output files:"
tot_return_val=0

for prpty in "pIC50_pred" "MPO" "molwt" "n_heavy_atoms"
do
    for suffix in "" "_by_substructure"
    do
        diff -sq hist_${prpty}${suffix}_ref.csv hist_${prpty}${suffix}.csv
        let tot_return_val=${tot_return_val}+$?
    done
done

echo -n "zdiff on compressed invalid_inchis.inchi.gz: "
zdiff -sq invalid_inchis_ref.inchi.gz invalid_inchis.inchi.gz
let tot_return_val=${tot_return_val}+$?

# Compare numeric and non-numeric columns of predictions file:

python <<EOF
import sys
import pandas as pd

df2 = pd.read_csv('all_preds_0-100.csv.gz', 
                  sep=';')

# Open correct reference results depending on whether test was run with Lilly 
# rules or not:
if 'Lilly_rules_pass' in df2.columns:
    all_preds_ref_file = 'all_preds_0-100_ref.csv.gz'
else:
    all_preds_ref_file = 'all_preds_0-100_ref_nolilly.csv.gz'
df1 = pd.read_csv(all_preds_ref_file, 
                  sep=';')

# Separate DataFrame into numeric and non-numeric types:

df1_numeric = df1.apply(pd.to_numeric, errors='ignore')\
                 .select_dtypes(include=[int, float])
df2_numeric = df2.apply(pd.to_numeric, errors='ignore')\
                 .select_dtypes(include=[int, float])

df1_str = df1[[col for col in df1.columns 
               if col not in df1_numeric.columns]]\
             .drop_duplicates()
df2_str = df2[[col for col in df2.columns 
               if col not in df2_numeric.columns]]\
             .drop_duplicates()

# Remove time columns:
df1_numeric = df1_numeric[[col for col in df1_numeric.columns 
                           if col[-5:] != '_time']]
df2_numeric = df2_numeric[[col for col in df2_numeric.columns 
                           if col[-5:] != '_time']]

# Compare DataFrames:

test_pass = True

# Compare string columns:

if len(pd.concat([df1_str, df2_str]).drop_duplicates()) > \
    max([len(df1_str), len(df2_str)]):
    print('Non-numeric columns of files {} and all_preds_0-100.csv.gz differ'\
          .format(all_preds_ref_file))
    print(df1_str.compare(df2_str, 
                          align_axis=0, 
			  #result_names=("ref", "new")
                         ))
    test_pass = False
else:
    print('Non-numeric columns of files {} and all_preds_0-100.csv.gz are '\
          'identical'.format(all_preds_ref_file))

# Compare numeric columns:

tolerance = 0.001

#print('Results difference tolerance:', tolerance)

diff_below_tol = bool((df1_numeric - df2_numeric).max().max() < tolerance)

if not diff_below_tol:
    print('Numeric columns of files {} and all_preds_0-100.csv.gz differ by '\
          'more than tolerance (> {})'.format(all_preds_ref_file, tolerance))
    print('Difference in numeric columns:')
    print(df1_numeric - df2_numeric)

    test_pass = False
else:
    print('Numeric columns of files {} and all_preds_0-100.csv.gz are within '\
          'tolerance (< {})'.format(all_preds_ref_file, tolerance))

# Set return value:
if not test_pass:
    sys.exit(1)
sys.exit(0)
EOF
let tot_return_val=${tot_return_val}+$?

# Delete output files:
rm all_preds_0-100.csv.gz
rm invalid_inchis.inchi.gz
rm mol_counts.csv
for prpty in "pIC50_pred" "MPO" "molwt" "n_heavy_atoms"
do
    for suffix in "" "_by_substructure"
    do
        rm hist_${prpty}${suffix}.csv
    done
done

echo "Number of files which differ: ${tot_return_val}"

if [ ${tot_return_val} == 0 ]
then
    echo "==========="
    echo "Test passed"
    echo "==========="
    exit 0
else
    echo "==========="
    echo "Test failed"
    echo "==========="
    exit 1
fi
