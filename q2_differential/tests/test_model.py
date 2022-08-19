import unittest
import argparse
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
# TODO move over to pytest


#parser = argparse.ArgumentParser()                                          
#parser.add_argument(                                                        
#     '--biom-table', help='Biom table of counts.', required=True)            
#parser.add_argument(                                                        
#     '--metadata-file', help='Sample metadata file.', required=True)         
#args = parser.parse_args()                                                  
#print(args.biom_table)                                                      
#table = biom.load_table(args.biom_table)                                    
#metadata = pd.read_table(args.metadata_file, index_col=0)

class TestDiseaseSingle(unittest.TestCase):
    def setUp(self):
        #self.table = table
        #self.metadata = metadata
        self.table = biom.load_table(get_data_path('/mnt/home/djin/ceph/snakemake/data/Dan2020ASD_rl150/tenMicrobes.biom'))
        # TODO change to relative paths
        self.metadata = pd.read_table(get_data_path('/mnt/home/djin/ceph/snakemake/data/Dan2020ASD_rl150/metadata_simple_fake.txt'),       
                              index_col=0)
        #unittest does not work with argparse
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--biom-table', help='Biom table of counts.', required=True)
    # parser.add_argument(
    #     '--metadata-file', help='Sample metadata file.', required=True)
    # args = parser.parse_args()
    # print(args.biom_table)
    # table = biom.load_table(args.biom_table)
    # metadata = pd.read_table(args.metadata_file, index_col=0)
    unittest.main()
