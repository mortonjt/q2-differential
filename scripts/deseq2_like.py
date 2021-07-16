import argparse
from biom import load_table
import pandas as pd
import numpy as np
import xarray as xr
from q2_differential._model import DESeq2
import time
import logging
import subprocess, os
import tempfile
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
        '--output-inference', help='Output inference tensor.',
        type=str, required=True)

    args = parser.parse_args()
    print(args)
    table = biom.load_table(get_data_path('table.biom'))
    metadata = pd.read_table(get_data_path('sample_metadata.txt'), index_col=0)
    model = DESeq2(table, metadata, args.groups,
                   num_iter=args.monte_carlo_chains,
                   num_warmup=args.monte_carlo_chains)
    model.compile_model()
    model.fit_model()
    samples = model.to_inference_object()
    samples.to_netcdf(args.output_inference)
