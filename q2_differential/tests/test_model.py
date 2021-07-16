import unittest
import biom
import pandas as pd
import numpy as np
from skbio.util import get_data_path
from q2_differential._model import DESeq2
from xarray.ufuncs import log2 as xlog
import pandas.testing as pdt


class TestDEseq2(unittest.TestCase):
    def setUp(self):
        self.table = biom.load_table(get_data_path('table.biom'))
        self.metadata = pd.read_table(get_data_path('sample_metadata.txt'), index_col=0)

    def test_stan_run(self):
        model = DESeq2(self.table, self.metadata, 'Status',
                       num_iter=128, num_warmup=1000)
        model.compile_model()
        model.fit_model()
        inf = model.to_inference_object()


class TestALDEx2(unittest.TestCase):
    def setUp(self):
        self.truth = pd.read_csv(get_data_path('aldex2-fit.csv'), index_col=0)
        self.table = biom.load_table(get_data_path('table.biom'))
        self.metadata = pd.read_table(get_data_path('sample_metadata.txt'), index_col=0)

    @unittest.skip('This implementation of ALDEx2 is not '
                   'correctly parameterized.')
    def test_stan_run(self):
        D = self.table.shape[0]
        model = ALDEx2(self.table, self.metadata, 'Status',
                       num_iter=128, num_warmup=1000)
        model.compile_model()
        model.fit_model()
        inf = model.to_inference_object()
        # # extract log fold change
        # beta = xlog(inf['posterior'])
        # # convert beta to clr coordinates
        # denom = beta.mean(dim=['feature'])
        # beta = beta - denom
        # asd = beta.loc[dict(group='ASD')]
        # con = beta.loc[dict(group='Control')]
        # res_lfc = (con - asd).median(dim=['chain', 'draw'])
        # res_lfc = res_lfc.to_pandas().sort_index()['beta']
        # exp_lfc = truth['diff.btw'].sort_index()



if __name__ == '__main__':
    unittest.main()
