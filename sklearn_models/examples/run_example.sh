#! /bin/bash

# Run example using parameters in run_params.json file and
# with random seed set to 0 to make results reproducible:

#python ../scripts/run_sklearn_models.py run_params.json 0
python ../../deepchem_models/scripts/run_deepchem_models.py run_params.json 0
