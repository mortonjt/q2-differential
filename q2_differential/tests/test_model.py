import unittest
import argparse
import biom
from biom.table import Table
import pandas as pd
import numpy as np
import numpy.testing as npt
from birdman.model_util import concatenate_inferences
from birdman import ModelIterator
from skbio.util import get_data_path
from xarray.ufuncs import log2 as xlog
import pandas.testing as pdt
from q2_differential._model import DESeq2, SingleDESeq2
from q2_differential._model import DiseaseSingle, _single_func
from q2_differential._stan import _case_control_negative_binomial_sim
from q2_differential._matching import _matchmaker
from skbio.stats.composition import alr_inv, clr
from multiprocessing import Pool
from q2_differential._model import _swap, DiseaseSingle
import pytest


class TestDiseaseSingle(unittest.TestCase):

    def setUp(self):
        t1, m1, d1 = _case_control_negative_binomial_sim(
            n=40, d=4, depth=100, diff_scale=1)
        t2, m2, d2 = _case_control_negative_binomial_sim(
            n=40, d=4, depth=100, diff_scale=2)
        m2.loc[m2['cc_bool'] == 1, 'cc_bool'] = 2
        # have to rename samples, since the names are identical
        m2.index = list(map(lambda x: f'{x}_2', m2.index))
        self.table = pd.concat((t1, t2), axis=0)
        self.metadata = pd.concat((m1, m2), axis=0)
        self.diff = pd.DataFrame({
            'diff1': d1,
            'diff2': d2
        })

    def test_swap(self):
        x = np.array([1,1,1,2,2,2,2,2,2,0,0,0])
        y = _swap(x, 0, 2)
        expx = np.array([1,1,1,0,0,0,0,0,0,2,2,2])
        npt.assert_allclose(y, expx)

    def test_sim(self):
        biomT = biom.Table(
            self.table.T.values, self.table.columns, self.table.index)

        models = ModelIterator(
            biomT, DiseaseSingle, metadata=self.metadata,
            category_column='cc_bool',
            match_ids_column='cc_ids',
            batch_column='batch_ids',
            reference=0,
            chains=4,
            num_iter=100,
            num_warmup=100)
    coords = {'feature' : biomT.ids(axis='observation')}
    posterior = concatenate_inferences(samples, coords, 'feature')


if __name__ == '__main__':
    unittest.main()
