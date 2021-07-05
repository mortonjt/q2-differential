import unittest
import biom
import pandas as pd
import numpy as np
from skbio.util import get_data_path
from q2_differential._model import ALDEx2


class TestDEseq2(unittest.TestCase):
    def setUp(self):

        pass


class TestALDEx2(unittest.TestCase):
    def setUp(self):
        self.truth = pd.read_table(get_data_path('aldex2-fit.csv'), index_col=0)
        self.table = biom.load_table(get_data_path('table.biom'))
        self.metadata = pd.read_table(get_data_path('sample_metadata.txt'), index_col=0)

    def test_stan_run(self):
        D = self.table.shape[0]
        model = ALDEx2(self.table, 'Status', self.metadata)
        model.compile_model()
        model.fit_model()
        inf = model.to_inference_object()


if __name__ == '__main__':
    unittest.main()
