import pandas as pd

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
