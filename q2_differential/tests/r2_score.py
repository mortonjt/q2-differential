#!/bin/bash

import arviz as az
import numpy as np
import biom
data_dir = '/mnt/home/djin/ceph/snakemake/data/test_Dan2020_Franzosa2019CD'
inference = az.InferenceData.from_netcdf(f'{data_dir}/differential_0829.nc')
table = biom.load_table(f'{data_dir}/Dan_Fran_CD.biom')
y_obs = table.matrix_data.todense().T.astype(int)
print(y_obs)
y_pred = inference.posterior_predictive.stack(sample=("chain","draw"))
y_pred = y_pred.stack(xfeature=('feature','tbl_sample'))['y_predict']
print(y_pred)
y_pred = y_pred.fillna(0)
r2_s = az.r2_score(y_pred.values, np.ravel(y_obs))
print(r2_s)
