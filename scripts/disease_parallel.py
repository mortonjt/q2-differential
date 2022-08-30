import argparse
import biom
import pandas as pd
import numpy as np
import xarray as xr
from q2_differential._model import DiseaseSingle, _single_func
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
                          '(i.e. treatment vs control groups).'),required=True)
    parser.add_argument(
        '--disease-column', help='The name of the disease group.', required=True)
    parser.add_argument(
        '--batch-column', help='The name of the batch column.', required=True)
    parser.add_argument(
        '--match-ids', help='The name of the match ids.', required=True)
    parser.add_argument(
        '--reference', help='The name of the control group.', required=True)
    parser.add_argument(
        '--monte-carlo-samples', help='Number of monte carlo samples.',
        type=int, required=False, default=100)
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
    DiseaseSingle(table, metadata=metadata,
                  feature_id=table.ids(axis='observation')[0],
                  category_column=args.disease_column,
                  batch_column=args.batch_column,
                  reference=args.reference,
                  match_ids_column=args.match_ids).compile_model()
    # Groups should be 0, 1, 2...?
    models = ModelIterator(table, DiseaseSingle, metadata=metadata,
                           category_column=args.disease_column,
                           match_ids_column=args.match_ids,
                           batch_column=args.batch_column,
                           reference=args.reference,
                           chains=args.chains,
                           num_iter=args.monte_carlo_samples,
                           num_warmup=1000)

    samples = []
    with Pool(args.processes) as p:
        for inf in p.imap(_single_func, models, chunksize=50):
            samples.append(inf)
    coords = {'feature' : table.ids(axis='observation')}
    samples = concatenate_inferences(samples, coords, 'feature')
    samples.to_netcdf(args.output_inference)
