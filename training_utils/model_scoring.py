import pandas as pd
import os

from training_utils.record_results import preds_file_index_cols

def get_average_model_performance(df_final_results, mode='regression', ext_test_sets=[]):

    if mode == 'regression':
        # Get average, standard deviation and standard error:
        df_av_results = pd.DataFrame()

        for dataset in ['train', 'test'] + ext_test_sets:
            df_agg = df_final_results[dataset].agg(['mean', 'std', 'sem']).T
            df_agg.index.rename('metric', inplace=True)

            # Calculate confidence intervals:
            df_agg.loc['rmsd', 'CI95_lower'] = df_agg.loc['rmsd', 'mean']-1.69*df_agg.loc['rmsd', 'sem']
            df_agg.loc['rmsd', 'CI95_upper'] = df_agg.loc['rmsd', 'mean']+1.69*df_agg.loc['rmsd', 'sem']
        
            df_agg['dataset'] = dataset
            df_agg = df_agg.set_index('dataset', append=True)\
                           .reorder_levels(['dataset', 'metric'])
        
            df_av_results = df_av_results.append(df_agg)
        
        return df_av_results


# Originally from run_deepchem_models script:
def score_ensemblelike_model_from_resamples(run_input,
                                            run_results={},
                                            preds_filename='',
                                            all_metrics={}):
    """
    Get average predictions over resamples and calculate stats.
    """

    # Average refit predictions and then calculate performance:
    if run_input["training"].get("calculate_ensemble_performance") and \
       run_input["training"].get("save_predictions") in ["refit", "all"]:
        if not os.path.isfile(preds_filename):
            print('WARNING: Cannot generate ensemble-like model if predictions not saved.')
            return
        print('Calculating performance of "ensemble" of refit models')
        df_preds = pd.read_csv(preds_filename, index_col=list(range(len(preds_file_index_cols))))

        df_av_preds = df_preds.loc[(df_preds.index.get_level_values('cv_fold') == 'refit') & \
                                   (df_preds.index.get_level_values('data_split') == 'test')]\
                              .reset_index(level='task')\
                              .groupby('task')\
                              .mean()

        df_av_preds.index = pd.MultiIndex.from_tuples([(-1, 'av_over_refits', -1, 'test',
                                                        task) for task in df_av_preds.index])

        # Save average predictions to file:
        df_av_preds.to_csv(preds_filename, mode='a', header=False)

        df_av_preds.dropna(axis=1, inplace=True)

        train_vals = pd.read_csv(run_input["dataset"]["dataset_file"])\
                       .set_index(run_input["dataset"]["id_field"])\
                       .loc[df_av_preds.columns, run_input["dataset"]["tasks"][0]]\
                       .to_numpy()

        for metric_name in all_metrics['_order']:
            metric = all_metrics[metric_name]
            run_results[('test', metric.name)] = round(metric.compute_metric(
                train_vals,
                df_av_preds.loc[(-1, 'av_over_refits', -1, 'test',
                                 run_input["dataset"]["tasks"][0])].to_numpy()), 3)

        # Save stats to file:
        # Maintain header order:
        df_ens_model_result = pd.read_csv('GCNN_info_refit_models.csv',
                                          sep=';', header=[0, 1], nrows=0)
        df_ens_model_result = df_ens_model_result.append(run_results.to_frame()\
                                                                    .T)\
                                                 [df_ens_model_result.columns]
        df_ens_model_result.to_csv('GCNN_info_refit_models.csv', header=False,
                                   mode='a', index=False, sep=';')
