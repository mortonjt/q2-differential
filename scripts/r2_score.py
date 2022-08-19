#!/bin/bash
import arviz as az
import numpy as np
import biom
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--biom-table', help='Biom table of counts.', required=True)
    parser.add_argument(
        '--nc-file', help='differential abundance file.', required=True)
    args = parser.parse_args()
    print(args)
    biom = biom.load_table(args.biom_table)
    nc = az.InferenceData.from_netcdf(args.nc_file)

#inference = az.InferenceData.from_netcdf('test.nc')
#table = biom.load_table('../q2_differential/tests/data/table36new.biom')
inference = nc
table = biom
y_obs = table.matrix_data.todense().T.astype(int)
y_pred = inference.posterior_predictive.stack(sample=("chain","draw"))
y_pred = y_pred.stack(xfeature=('feature','tbl_sample'))['y_predict']
r2_s = az.r2_score(y_pred.values, np.ravel(y_obs))
print(r2_s)
