import os
import numpy as np
import pandas as pd
from skbio.stats.composition import (closure, alr, alr_inv,
                                     multiplicative_replacement)
from sklearn.preprocessing import LabelEncoder
from cmdstanpy import CmdStanModel, CmdStanMCMC
import tempfile
import json
import xarray as xr
import arviz as az
from scipy.stats import nbinom
from q2_differential._matching import _matchmaker

#https://github.com/flatironinstitute/q2-matchmaker/blob/main/q2_matchmaker/_stan.py#L72

def negative_binomial_rvs(mu, alpha, state=None):
    """ Uses mean / phi reparameterization of scipy negative binomial"""

    sigma2 = mu + alpha * mu ** 2
    p = mu / sigma2
    n = (mu ** 2) / (sigma2 - mu)
    return nbinom.rvs(n, p, random_state=state)


def _case_control_negative_binomial_sim(n=100, b=2, d=10, depth=50,
                                        disp_scale = 0.1,
                                        batch_scale = 0.1,
                                        diff_scale = 1,
                                        control_loc = None,
                                        control_scale = 0.1,
                                        state=None, params=dict()):
    """ Simulate case-controls from Negative Binomial distribution
    Parameters
    ----------
    n : int
       Number of samples (must be divisible by 2).
    b : int
       Number of batches (must be able to divide n).
    d : int
       Number of microbes
    depth : int
       Sequencing depth
    state : np.random.RandomState or int or None
       Random number generator.
    params : dict
       Dictionary of parameters to initialize simulations
    Returns
    -------
    table : pd.DataFrame
        Simulated counts
    md : pd.DataFrame
        Simulated metadata
    diff : pd.DataFrame
        Ground truth differentials
    """
    if state is None:
        state = np.random.RandomState(0)
    else:
        state = np.random.RandomState(state)

    # dimensionality
    c = n // 2
    # setup scaling parameters
    if control_loc is None:
        control_loc = np.log(1 / d)
    eps = 0.1      # random effects for intercepts
    delta = 0.1    # size of overdispersion
    # setup priors
    a1 = state.normal(eps, eps, size=d)
    diff = params.get('diff', state.normal(0, diff_scale, size=d))
    disp = params.get('disp', state.lognormal(np.log(delta), disp_scale, size=(2, d)))

    batch_mu = params.get('batch_mu', state.normal(0, 1, size=(b, d)))
    batch_disp = params.get('batch_disp', state.lognormal(np.log(delta), batch_scale, size=(b, d)))
    control_mu = params.get('control_mu', state.normal(control_loc, 1, size=(d)))
    control_sigma = params.get('control_sigma', state.lognormal(np.log(delta), control_scale, size=(d)))
    control = np.vstack([state.normal(control_mu, control_sigma) for _ in range(c)])

    depth = np.log(state.poisson(depth, size=n))
    # depth = np.array([np.log(depth)] * n)  # for debugging
    # look up tables
    bs = n // b  # batch size
    batch_ids = np.repeat(np.arange(b), bs)
    batch_ids = np.hstack((
        batch_ids,
        np.array([b - 1] * (n - len(batch_ids)))
    )).astype(np.int64)
    cc_bool = np.arange(n) % 2  # healthy or disease1 or disease2
    cc_ids = np.repeat(np.arange(c), 2)
    y = np.zeros((n, d))
    # model simulation
    #TO DO: modify this to simulate multiple diseases
    for s in range(n):
        for i in range(d):
            # control counts
            lam = depth[s] + batch_mu[batch_ids[s], i] + control[cc_ids[s], i]
            # case counts (if applicable)
            if cc_bool[s] == 1:
                lam += diff[i]
            alpha = (np.exp(a1[i]) / np.exp(lam))
            alpha += disp[cc_bool[s], i]
            alpha += batch_disp[batch_ids[s], i]
            # phi = (1 / alpha)  # stan's parameterization
            nb = negative_binomial_rvs(np.exp(lam), alpha, state)
            y[s, i] = nb
    oids = [f'o{x}' for x in range(d)]
    sids = [f's{x}' for x in range(n)]
    table = pd.DataFrame(y, index=sids, columns=oids)
    #TO DO: add match_ids_column, batch_column, and reference
    md = pd.DataFrame({'cc_bool': cc_bool.astype(np.str),
                       'cc_ids': cc_ids.astype(np.str),
                       'batch_ids': batch_ids.astype(np.str)},
                       #'match_ids_column':
                       #'batch_column':
                       #'reference':},
                      index=sids)
    md.index.name = 'sample id'
    return md
metadata = pd.read_table('/mnt/home/djin/ceph/snakemake/data/Qin2010IBD/Qin2010IBD_metadata.txt',
                              index_col=0)
