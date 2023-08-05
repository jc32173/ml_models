run_input = \
{
    "model_info" : {
        "notes" : "Test run..."
    },
    "dataset" : {
        "mode" : "regression",
        "dataset_file" : "../../example_data/solubility_challenge_loose_set.csv",
        "tasks" : ["Solubility"],
        "feature_field" : "SMILES",
        "id_field" : "ID",
        "descriptors" : ["RDKit"],
	"fingerprints" : ["Morgan"]
    },
    "ext_datasets" : {
	"loose" : {"dataset_file" : "../../example_data/solubility_challenge_loose_set.csv"
	},
	"tight" : {"dataset_file" : "../../example_data/solubility_challenge_loose_set.csv"
        },
        "_order" : ["loose", "tight"]
    },
    "preprocessing" : {
        "ph" : 7.4,
        "phmodel" : "OpenEye",
        "tauto" : True
    },
    "train_test_split" : {
	"split_method" : "random",
	"n_splits" : 2,
        "frac_train" : 0.7,
        "frac_test" : 0.3
    },
    "train_val_split" : {
        "split_method" : "k-fold",
	"n_splits" : 2
    },
    "training" : {
	"n_cpus" : 2,
        "model_fn_str" : "RmCorrVar_RFR",
        "hyperparam_search" : "random",
	"n_iter" : 5,
	"save_model" : "resample",
	"save_predictions" : False
    },
    "hyperparams" : {
	"RmCorrVar__corr_cutoff" : [0.9, 0.95],
        "RFR__n_estimators" : [20, 40],
        "RFR__max_depth" : [5, 10],
        "RFR__min_samples_split" : [2, 4],
        "RFR__min_samples_leaf" : [1, 2],
	"_order" : ["RmCorrVar__corr_cutoff", "RFR__n_estimators", "RFR__max_depth", "RFR__min_samples_split", "RFR__min_samples_leaf"]
    }
}
