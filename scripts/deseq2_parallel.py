import argparse
import biom
import pandas as pd
import numpy as np
import xarray as xr
from q2_differential._model import SingleDESeq2
from birdman.model_util import concatenate_inferences
from birdman import ModelIterator
import time
import logging
import subprocess, os
import tempfile
from multiprocessing import Pool
import arviz as az


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--biom-table', help='Biom table of counts.', required=True)
    parser.add_argument(
        '--metadata-file', help='Sample metadata file.', required=True)
    parser.add_argument(
        '--groups', help=('Column specifying groups '
                          '(i.e. treatment vs control groups).'),
        required=True)
    parser.add_argument(
        '--control-group', help='The name of the control group.', required=True)
    parser.add_argument(
        '--monte-carlo-samples', help='Number of monte carlo samples.',
        type=int, required=False, default=1000)
    parser.add_argument(
        '--chains', help='Number of MCMC chains.', type=int,
        required=False, default=4)
    parser.add_argument(
        '--processes', help='Number of parallel processes to run.', type=int,
        required=False, default=1)
    parser.add_argument(
        '--output-inference', help='Output inference tensor.',
        type=str, required=True)
    args = parser.parse_args()
    print(args)
    table = biom.load_table(args.biom_table)
    metadata = pd.read_table(args.metadata_file, index_col=0)
    # initialize just to compile model
    SingleDESeq2(table, metadata, 'Status').compile_model()

    models = ModelIterator(table, SingleDESeq2, metadata=metadata,
                           category_column='Status', chains=args.chains,
                           num_iter=args.monte_carlo_samples,
                           num_warmup=args.monte_carlo_samples)

    def _single_func(x):
        fid, m = x
        m.fit_model()
        return m.to_inference_object()

    samples = []
    with Pool(args.processes) as p:
        for inf in p.imap(_single_func, models, chunksize=10):
            samples.append(inf)
    coords = {'feature' : table.ids(axis='observation')}
    samples = concatenate_inferences(samples, coords, 'feature')
    samples.to_netcdf(args.output_inference)
