#!/bin/bash
import arviz as az
import numpy as np
import biom
inference = az.InferenceData.from_netcdf('test.nc')
table = biom.load_table('../q2_differential/tests/data/table36new.biom')
y_obs = table.matrix_data.todense().T.astype(int)
y_pred = inference.posterior_predictive.stack(sample=("chain","draw"))
y_pred = y_pred.stack(xfeature=('feature','tbl_sample'))['y_predict']
r2_s = az.r2_score(y_pred.values, np.ravel(y_obs))
print(r2_s)
