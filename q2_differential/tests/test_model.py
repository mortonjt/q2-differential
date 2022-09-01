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
        pass

    def test_swap(self):
        x = np.array([1,1,1,2,2,2,2,2,2,0,0,0])
        y = _swap(x, 0, 2)
        expx = np.array([1,1,1,0,0,0,0,0,0,2,2,2])
        npt.assert_allclose(y, expx)


    def test_sim(self):
        pass

if __name__ == '__main__':
    unittest.main()
