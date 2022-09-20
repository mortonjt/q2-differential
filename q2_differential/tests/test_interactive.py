import arviz as az
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


d = 10
delta = 0.1
control_loc = np.log(1/d)
control_scale = 1
state = np.random.RandomState(0)
params = {
    'control_mu': state.normal(control_loc, 1, size=(d)),
    'control_sigma': state.lognormal(np.log(delta), control_scale, size=(d))
}
t1, m1, d1 = _case_control_negative_binomial_sim(
    n=200, d=d, b=1, depth=1000, diff_scale=1, params=params, state=state)
#t2, m2, d2 = _case_control_negative_binomial_sim(
#    n=40, d=d, b=1, depth=1000, diff_scale=1, params=params, state=state)
# rename diseases
#m2.loc[m2['cc_bool'] == '1', 'cc_bool'] = '2'
# m2.index = list(map(lambda x: f'{x}_2', m2.index))
# t2.index = list(map(lambda x: f'{x}_2', t2.index))
#
# table = pd.concat((t1, t2), axis=0)
# metadata = pd.concat((m1, m2), axis=0)
table = t1
metadata = m1

# expected differences
# exp_diffs = pd.DataFrame({'1': d1,
#                           '2': d2}, index=table.columns)
exp_diffs = pd.DataFrame({'1': d1}, index=table.columns)

# align up with results
exp_diffs = pd.melt(exp_diffs.reset_index(), id_vars='index')
exp_diffs = (exp_diffs
             .rename(columns={'index': 'feature',
                              'variable': 'disease_ids',
                              'value': 'diff'})
             .set_index(['feature', 'disease_ids'])['diff']
             .sort_index())

biomT = biom.Table(table.T.values, table.columns, table.index)

models = ModelIterator(
    biomT, DiseaseSingle, metadata=metadata,
    category_column='cc_bool',
    match_ids_column='cc_ids',
    batch_column='batch_ids',
    reference='0',
    chains=4,
    num_iter=100,
    num_warmup=1000)
coords = {'feature' : biomT.ids(axis='observation')}

samples = []

for m in models:
    samples.append(_single_func(m))

posterior = concatenate_inferences(samples, coords, 'feature')

# First compute R2 to see if the fit is good
y_true = table.values.ravel()

pp = posterior.posterior_predictive
# Assume only one data variable
pp_name = list(pp.data_vars)[0]
y_pred = pp[pp_name].stack(mcmc_sample=["chain", "draw"])
y_pred = y_pred.stack(entry=["tbl_sample", "feature"]).data
print(az.r2_score(y_true, y_pred))


res_diff_5 = (posterior['posterior']['diff']
              .to_dataframe()
              .reset_index()[['feature', 'disease_ids', 'diff']]
              .groupby(['feature', 'disease_ids'])
              .quantile(0.05)
              .reset_index()
              .query("disease_ids == '1'")
              .set_index(['feature', 'disease_ids'])['diff']
              .sort_index())
res_diff_95 = (posterior['posterior']['diff']
               .to_dataframe()
               .reset_index()[['feature', 'disease_ids', 'diff']]
               .groupby(['feature', 'disease_ids'])
               .quantile(0.95)
               .reset_index()
               .query("disease_ids == '1'")
               .set_index(['feature', 'disease_ids'])['diff']
               .sort_index())

# check to see if the majority of ground truth log-fold changes are within
# the 95% confidence intervals
assert np.mean(res_diff_5.values < exp_diffs.values) >= 0.9
assert np.mean(res_diff_95.values > exp_diffs.values) >= 0.9
