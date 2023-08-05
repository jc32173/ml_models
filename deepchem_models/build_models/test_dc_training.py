import unittest
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import deepchem as dc
from io import StringIO

from deepchem_models.build_models.dc_training import save_predictions_to_file

class toy_model():
    """
    Toy predictive model for testing with a predict method 
    that takes a dataset and returns the y data unmodified.
    """
    def predict(self, dataset):
        return dataset.y


class Test_save_predictions_to_file(unittest.TestCase):

    def setUp(self):
        smiles = np.array(['CCCC', 'CC(=O)OCC', 'COCCC', 'C'])
        y = np.array([[2], [7], [4], [3]])
        ids = np.array(['mol0', 'mol1', 'mol2', 'mol3'])
        
        self.ids = ids
        self.model = toy_model()

        featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
        feats = featurizer.featurize(smiles)
        train_set_idxs = [0, 2]
        val_set_idxs = [1]
        test_set_idxs = [3]
        self.train_set = dc.data.NumpyDataset(X=feats[train_set_idxs],
                                              y=y[train_set_idxs],
                                              ids=ids[train_set_idxs])
        self.val_set = dc.data.NumpyDataset(X=feats[val_set_idxs],
                                              y=y[val_set_idxs],
                                              ids=ids[val_set_idxs])
        self.test_set = dc.data.NumpyDataset(X=feats[test_set_idxs],
                                              y=y[test_set_idxs],
                                              ids=ids[test_set_idxs])

    # Use mock to override isfile test when using StringIO rather than file:
    @unittest.mock.patch('os.path.isfile')
    def test_output(self, mock_isfile):
        """
        Check the predictions output is the correct format.
        """

        mock_isfile.return_value = True

        df_preds = pd.DataFrame(
            data=[], 
            columns=['resample_number', 'cv_fold', 'model_number', 'data_split'] + \
                    self.ids.tolist(), 
            index=[])

        #df_ext_preds = pd.DataFrame(
        #    data=[], 
        #    columns=['resample_number', 'cv_fold', 'model_number', 'data_split'] + \
        #            self.ids.tolist(), 
        #    index=[])

        preds_file = StringIO()
        ext_preds_file = StringIO()

        df_preds.to_csv(preds_file, index=False)
        preds_file.seek(0)
        ext_preds_file = None

        save_predictions_to_file(model=self.model,
                                 resample_number=1,
                                 cv_fold=3,
                                 model_number=15,
                                 train_set=self.train_set,
                                 val_set=self.val_set,
                                 test_set=self.test_set,
                                 ext_test_set={},
                                 transformers=[],
                                 preds_file=preds_file,
                                 ext_preds_file=ext_preds_file,
                                )

        preds_file.seek(0)
        df_preds = pd.read_csv(preds_file, index_col=[0, 1, 2, 3])

        correct_output = pd.DataFrame(
            data=[{molid : pred for molid, pred in zip(dataset.ids, dataset.y)} 
                  for dataset in [self.train_set, self.val_set, self.test_set]],
            index=pd.MultiIndex.from_tuples(
                [tuple([1, 3, 15] + [split]) for split in ['train', 'val', 'test']], 
                names=['resample_number', 'cv_fold', 'model_number', 'data_split']), 
            columns=self.ids)

        pd_testing.assert_frame_equal(df_preds, correct_output)
