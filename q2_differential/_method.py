import qiime2
import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
import biom
from q2_differential._stan import (
    _case_control_full, _case_control_data,
    _case_control_single)
from q2_differential._matching import _matchmaker
from typing import List


def negative_binomial_case_control(
        table: pd.DataFrame,
        matching_ids: qiime2.CategoricalMetadataColumn,
        groups: qiime2.CategoricalMetadataColumn,
        monte_carlo_samples: int = 2000,
        reference_group : str = 'TD',
        cores : int = 1) -> az.InferenceData:

    metadata = pd.DataFrame({'cc_ids': matching_ids.to_series(),
                             'groups': groups.to_series()})
    metadata['groups'] = (metadata['groups'] == reference_group).astype(np.int64)

    # take intersection
    idx = list(set(metadata.index) & set(table.index))
    counts = table.loc[idx]
    metadata = metadata.loc[idx]
    depth = counts.sum(axis=1)
    dat = _case_control_data(counts.values,
                             metadata['cc_ids'].values,
                             metadata['groups'].values, depth)
    _, posterior, prior = _case_control_full(
        counts=counts.values,
        case_ctrl_ids=metadata['cc_ids'].values,
        case_member=metadata['groups'].values,
        depth=depth,
        mc_samples=monte_carlo_samples)
    opts = {
        'observed_data': dat,
        'coords': {'diff': list(table.columns[1:])}
    }
    samples = az.from_cmdstanpy(posterior=posterior, prior=prior, **opts)
    return samples


def parallel_negative_binomial_case_control(
        table: pd.DataFrame,
        matching_ids: qiime2.CategoricalMetadataColumn,
        groups: qiime2.CategoricalMetadataColumn,
        monte_carlo_samples: int = 2000,
        reference_group : str = 'TD',
        cores : int = 1) -> xr.Dataset:

    metadata = pd.DataFrame({'cc_ids': matching_ids.to_series(),
                             'groups': groups.to_series()})
    metadata['groups'] = (metadata['groups'] == reference_group).astype(np.int64)

    # take intersection
    idx = list(set(metadata.index) & set(table.index))
    counts = table.loc[idx]
    metadata = metadata.loc[idx]
    depth = counts.sum(axis=1)
    pfunc = lambda x: _case_control_single(
        counts=np.array(x.values),
        case_ctrl_ids=metadata['cc_ids'].values,
        case_member=metadata['groups'].values,
        depth=depth,
        mc_samples=monte_carlo_samples)
    if cores > 1:
        try:
            import dask.dataframe as dd
            dcounts = dd.from_pandas(counts.T, npartitions=cores)
            res = dcounts.apply(pfunc, axis=1)
            resdf = res.compute(scheduler='processes')
            data_df = list(resdf.values)
        except:
            data_df = list(counts.T.apply(pfunc, axis=1).values)
    else:
        data_df = list(counts.T.apply(pfunc, axis=1).values)
    samples = xr.concat([df.to_xarray() for df in data_df], dim="features")
    samples = samples.assign_coords(coords={
            'features' : counts.columns,
            'monte_carlo_samples' : np.arange(monte_carlo_samples)
    })
    return samples


def slurm_negative_binomial_case_control(
        table: pd.DataFrame,
        matching_ids: qiime2.CategoricalMetadataColumn,
        groups: qiime2.CategoricalMetadataColumn,
        reference_group : str,
        monte_carlo_samples: int = 2000,
        cores : int = 4,
        processes : int = 4,
        nodes : int = 2,
        memory : str = '16GB',
        walltime : str = '01:00:00',
        queue : str = '') -> xr.Dataset:
    import dask
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    import dask.array as da
    import logging
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)
    cluster = SLURMCluster(cores=cores,
                           processes=processes,
                           memory=memory,
                           walltime=walltime,
                           interface='ib0',
                           env_extra=["export TBB_CXX_TYPE=gcc"],
                           queue=queue)
    cluster.scale(jobs=nodes)

    metadata = pd.DataFrame({'cc_ids': matching_ids.to_series(),
                             'groups': groups.to_series()})
    metadata['groups'] = (metadata['groups'] == reference_group).astype(np.int64)

    # take intersection
    idx = list(set(metadata.index) & set(table.index))
    counts = table.loc[idx]
    metadata = metadata.loc[idx]
    depth = counts.sum(axis=1)
    pfunc = lambda x: _case_control_single(
        x, case_ctrl_ids=metadata['cc_ids'],
        case_member=metadata['groups'],
        depth=depth, mc_samples=monte_carlo_samples)
    dcounts = da.from_array(counts.values.T, chunks=(counts.T.shape))
    res = []
    for d in range(dcounts.shape[0]):
        r = dask.delayed(pfunc)(dcounts[d])
        res.append(r)
    futures = dask.persist(*res)
    resdf = dask.compute(futures)
    data_df = list(resdf[0])
    samples = xr.concat([df.to_xarray() for df in data_df], dim="features")
    samples = samples.assign_coords(coords={
            'features' : counts.columns,
            'monte_carlo_samples' : np.arange(monte_carlo_samples)
    })
    return samples


def dirichlet_multinomial(
        table: biom.Table,
        groups: qiime2.CategoricalMetadataColumn,
        training_samples: qiime2.CategoricalMetadataColumn = None,
        percent_test_examples: float = 0.1,
        monte_carlo_samples: int = 2000,
        reference_group: str = None) -> xr.DataArray:
    # Perform train/test split
    groups = groups.to_series()
    if training_samples is None:
        idx = np.random.random(len(groups)) < percent_test_examples
    else:
        idx = training_samples == 'Test'
    train_samples = set(groups.loc[~idx].index)
    func = lambda v, i, m: i in train_samples
    train_table = table.filter(func, inplace=False, axis='sample')
    func = lambda v, i, m: i not in train_samples
    test_table = table.filter(func, inplace=False, axis='sample')
    cats = list(groups.value_counts().index)
    if reference_group is None:
        ref_idx = 0
        reference_group = cats[0]
    else:
        ref_idx = cats.index(reference_group)
    # Compute Multinomial probabilities
    D, N = train_table.shape
    C = len(cats)
    samples = np.zeros((C, D, monte_carlo_samples))
    for j, c in enumerate(cats):
        sample_set = set(groups.index[groups == c])
        func = lambda v, i, m: i in sample_set
        subtable = table.filter(func, inplace=False)
        group_mean = subtable.sum(axis='observation') + 1
        # Draw MCMC samples
        samples[j] = np.random.dirichlet(
            group_mean, size=monte_carlo_samples).T
    # Build x-array object
    diffs = np.log((samples / np.expand_dims(samples[ref_idx], 0)))
    idx = np.array([reference_group != c for c in cats])
    diffs = diffs[idx]
    cats.remove(reference_group)
    samples = xr.DataArray(
        diffs,
        dims=['contrasts', 'features', 'monte_carlo_samples'],
        coords=dict(
            contrasts=cats,
            features=train_table.ids(axis='observation'),
            monte_carlo_samples=np.arange(monte_carlo_samples)
        ),
        attrs=dict(
            description='Posterior samples of groups',
            reference=reference_group
        )
    )
    return samples


def matching(sample_metadata : qiime2.Metadata,
             status : str,
             match_columns : List[str],
             prefix : str = None) -> qiime2.Metadata:
    new_column = 'matching_id'
    columns = [sample_metadata.get_column(col) for col in match_columns]
    types = [isinstance(m, qiime2.CategoricalMetadataColumn) for m in columns]
    sample_metadata = sample_metadata.to_dataframe()
    match_ids = _matchmaker(sample_metadata, status, match_columns, types)
    new_metadata = sample_metadata.copy()
    new_metadata[new_column] = match_ids
    # drop any nans that may appear due to lack of matching
    new_metadata = new_metadata.dropna(subset=[new_column])
    new_metadata[new_column] = new_metadata[new_column].astype(
        np.int64).astype(str)
    if prefix is not None:
        new_metadata[new_column] = new_metadata[new_column].apply(
            lambda x: f'{prefix}_{x}')
    return qiime2.Metadata(new_metadata)
