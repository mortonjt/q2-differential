import unittest
import biom
import pandas as pd
import numpy as np
from skbio.util import get_data_path
from q2_differential._model import DESeq2, SingleDESeq2, DiseaseSingle
from birdman import ModelIterator
from xarray.ufuncs import log2 as xlog
import pandas.testing as pdt
from q2_differential._stan import _case_control_negative_binomial_sim
from q2_differential._matching import _matchmaker
from skbio.stats.composition import alr_inv, clr
from biom.table import Table
from multiprocessing import Pool
from birdman.model_util import concatenate_inferences
#class TestDEseq2(unittest.TestCase):
#    def setUp(self):
#        self.table = biom.load_table(get_data_path('table.biom'))
#        self.metadata = pd.read_table(get_data_path('sample_metadata.txt'),
#                                      index_col=0)
#
#    def test_stan_run(self):
#        model = DESeq2(self.table, self.metadata, 'Status',
#                       num_iter=128, num_warmup=1000)
#        model.compile_model()
#        model.fit_model()
#        inf = model.to_inference_object()
#
#
#class TestSingleDEseq2(unittest.TestCase):
#    def setUp(self):
#        self.table = biom.load_table(get_data_path('table.biom'))
#        self.metadata = pd.read_table(get_data_path('sample_metadata.txt'),
#                                      index_col=0)
#
#    def test_stan_run(self):
#        models = ModelIterator(self.table, SingleDESeq2, metadata=self.metadata,
#                               category_column='Status', num_iter=128, num_warmup=1000)
#        for fid, m in models:
#            m.compile_model()
#            m.fit_model()
#            m.to_inference_object()

#Create a biom table contains 6 features and 6 samples
#data = np.arange(36).reshape(6,6)
#sample_ids = sample_ids = ['S%d' % i for i in range(6)]
#observ_ids = ['O%d' % i for i in range(6)]
#sample_metadata = [{'sampleid': 'SRR7057621'}, {'sampleid': 'SRR7057622'},
#                   {'sampleid': 'SRR7057623'}, {'sampleid': 'SRR7057624'},
#                   {'sampleid': 'SRR7057625'}, {'sampleid': 'SRR7057653'}]
#observ_metadata = [{'taxonomy': ['Bacteria', 'Firmicutes']},
#                   {'taxonomy': ['Bacteria', 'Proteobacteria']},
#                   {'taxonomy': ['Bacteria', 'Bacteroidetes']},
#                   {'taxonomy': ['Bacteria', 'Firmicutes']},
#                   {'taxonomy': ['Bacteria', 'Firmicutes']},
#                   {'taxonomy': ['Bacteria', 'Firmicutes']}]
#table36 = Table(data, observ_ids, sample_ids, observ_metadata,
#                sample_metadata, table_id='Example Table')

class TestDiseaseSingle(unittest.TestCase):
    def setUp(self):
        self.table = biom.load_table(get_data_path('/mnt/home/djin/ceph/snakemake/data/ASD_MS.test/tenMicrobes.biom'))
        self.metadata = pd.read_table(get_data_path('/mnt/home/djin/ceph/snakemake/data/ASD_MS.test/Dan_n_iMSMS.txt'),       
                              index_col=0)

#class TestDiseaseSingle(unittest.TestCase):
#    def setUp(self):
#        np.random.seed(0)
#        self.table, self.metadata, self.diff = _case_control_negative_binomial_sim(
#            n=40, d=4, depth=100)
#        self.diff = clr(alr_inv(self.diff))
#        observation_ids = list(self.table.columns)
#        sample_ids = list(self.table.index)
 #       count = self.table.values.T
#        self.table = Table(count,observation_ids, sample_ids)
        #matchmaker control and disease
        #self.metadata = _matchmaker(self.metadata,cc_bool,True)
    def test_stan_run(self):
        models = ModelIterator(self.table, DiseaseSingle, metadata=self.metadata,
                               match_ids_column='match_ids_column',
                               batch_column='batch_column',reference='Healthy',
                               category_column='Status', num_iter=128, num_warmup=1000)
        def _single_func(x):
            fid, m = x
            m.compile_model()
            m.fit_model()
            return m.to_inference_object()

        samples = []
        for m in models:
            inf = _single_func(m)
            samples.append(inf)
        coords = {'feature' : self.table.ids(axis='observation')}
#        samples = concatenate_inferences(samples, coords, 'feature')
#        samples.to_netcdf('test2.nc')

if __name__ == '__main__':
    unittest.main()
