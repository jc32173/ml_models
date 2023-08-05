import numpy as np
import pandas as pd


def score_model(model,
                train_set=None,
                val_set=None,
                test_set=None,
                ext_test_set={},
                all_metrics={},
                mode='regression',
                n_classes=None,
                run_results={},
                run_preds=None,
                **kwargs):

    # Get predictions and save stats on dataset splits:
    df_preds = pd.DataFrame()
    for split_name, dataset in {'train' : train_set,
                                'val' : val_set,
                                'test' : test_set,
                                **ext_test_set}.items():

        if dataset is not None:
            y_pred = pd.Series(model.predict(dataset.X).squeeze(),
                               index=dataset.ids.squeeze(),
                               name=split_name)
            y_true = dataset.y.squeeze()

            #for metric_name, metric_fn in all_metrics.items():
            for metric_name in all_metrics['_order']:
                metric_fn = all_metrics[metric_name]
                run_results[(split_name, metric_name)] = round(metric_fn(y_true, y_pred), 3)

            if mode == 'regression':
                run_results[(split_name, 'y_stddev')] = dataset.y.std()

            if run_preds is not None:
                run_preds[split_name] = y_pred
#            df_preds = df_preds.append(y_pred)
#    print(df_preds)
