## JSON file options for SKLearn models:

- `model_info`
    - `notes` (*optional*): Notes on the model.
- `dataset`
    - `dataset_file`: Path to csv file containing dataset for training.
    - `mode`: `regression` (*default*) | `classification`
    - `tasks`: List of column(s) in csv file to predict.
    - `feature_field` : Name of column in csv file containing SMILES.
    - `id_field`: Name of column containing in csv file containing molecule IDs.
    - `descriptors`: `["RDKit"]` (*default*) List of descriptor sets to calculate.
    - `extra_desc`: List of columns in `dataset_file` csv file to be included as descriptors.

- `ext_datasets` (*optional*): Details for other external datasets to use for testing the model, but not for training.
    - `<Name of dataset>`
        - `dataset_file`: Path to csv file containing dataset.
        - `tasks`: List of column(s) in csv file to predict.
        - `feature_field` : Name of column in csv file containing SMILES.
        - `id_field`: Name of column containing in csv file containing molecule IDs.

- `preprocessing`: 
    - `ph`: pH to convert SMILES to before featurizing.
    - `phmodel`: `OpenEye` | `OpenBabel`
    - `tauto`: `RDKit`

- `train_test_split`: Options to split dataset into separate training/test sets (outer split of nested CV).
    - `split_method`: `random` | `k-fold` | `stratified` | `murcko` | `predefined_lipo`
        (`murcko`: Stratified split using murcko scaffold, )
    - `n_splits`: Number of train/test resamples.
    - `strat_field`: Field in dataset file to use for 
    - `frac_train`: (used for ) Fraction of the dataset to use as the training set.
    - `frac_test`: Fraction of the dataset to use as the test set.

- `train_val_split`: Options to split training set into separate training/validation sets (inner split of nested CV).
    - *Options are the same as for `train_test_split`.*

- `training`: Options to control training procedure.
    - `n_cpus`: Number of CPUs to use.
    - `model_fn_str`: Name of model function.
    - `hyperparam_search`: `random` | `grid` Hyperparameter search strategy.
    - `n_iter`: Number of 
    - `save_model`: `refit` | `final`
    - `save_predictions`: `refit` | `final`

- `hyperparams`: Hyperparameter names and possible values to search over during hyperparameter tuning.  These will be specific for each type of model.

<!--- `feature_selection`:

 `models`:
   - \<Model name>
       -Options
       - `save_model`: `all` | `refit` (*default*) | `false`
       	    Save checkpoint files for model, after...

   - `hyperparams`: List of hyperparameter names and possible values to search over during hyperparameter tuning.  These will be specific for each type of model.
 --->

## Output files:

- `train_val_test_split.csv`: File containing the assignment of compounds to different training/validation/test splits.
- `<model_name>_av_performance.csv`: Model performance averaged over resamples.
