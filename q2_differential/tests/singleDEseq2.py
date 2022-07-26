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

class TestSingleDEseq2(unittest.TestCase):
   def setUp(self):
       self.table = biom.load_table(get_data_path('table.biom'))
       self.metadata = pd.read_table(get_data_path('sample_metadata.txt'),
                                     index_col=0)

   def test_stan_run(self):
       models = ModelIterator(self.table, SingleDESeq2, metadata=self.metadata,
                              category_column='Status', num_iter=128, num_warmup=1000)
       for fid, m in models:
           m.compile_model()
           m.fit_model()
           m.to_inference_object()

if __name__ == '__main__':
    unittest.main()

