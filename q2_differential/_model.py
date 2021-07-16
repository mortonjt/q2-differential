import os
import biom
import numpy as np
import pandas as pd
import birdman
from scipy.stats.mstats import gmean
from birdman.model_base import TableModel
import warnings


class DESeq2(TableModel):
    """ A model to mimic DESeq2. """
    def __init__(self,
                 table: biom.table.Table,
                 metadata: pd.DataFrame,
                 category_column: str,
                 reference: str = None,
                 beta_s : float = 1,
                 alpha_s : float = 1,
                 num_iter: int = 500,
                 num_warmup: int = None,
                 normalization: str = 'depth',
                 chains: int = 4,
                 seed: float = 42):

        filepath =  os.path.join(os.path.dirname(__file__),
                                 'assets/deseq2_simple.stan')
        super().__init__(table=table,
                         model_path=filepath,
                         num_iter=num_iter,
                         num_warmup=num_warmup,
                         chains=chains,
                         seed=seed)
        cats = metadata[category_column]
        other = list(set(cats) - {reference})[0]
        if reference is None:
            reference = cats[0]
        cats = (cats.values != reference).astype(np.int64) + 1
        other = list(set(cats) - {reference})[0]

        # compute normalization
        if normalization == 'median_ratios':
            K = table.matrix_data.todense().T + 0.5
            Km = gmean(counts, axis=0)
            slog = np.log(np.median(K / Km, axis=1))
        elif normalization == 'depth':
            slog = np.log(table.sum(axis='sample'))
        else:
            raise ValueError('`normalization` must be specified.')

        control_loc = np.log(1. / len(table.ids(axis='observation')))
        control_scale = 5
        param_dict = {
            "slog": slog,
            "M": cats,
            "control_loc": control_loc,
            "control_scale": control_scale
        }
        self.add_parameters(param_dict)
        self.specify_model(
            params=["intercept", "beta", "alpha"],
            dims={
                "intercept": ["feature"],
                "beta": ["feature"],
                "alpha": ["group", "feature"],
                "log_lhood": ["tbl_sample", "feature"],
                "y_predict": ["tbl_sample", "feature"]
            },
            coords={
                "group": [reference, other],
                "feature": self.feature_names,
                "tbl_sample": self.sample_names
            },
            include_observed_data=True,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood"
        )


class ALDEx2(TableModel):
    """ Note that this is work in progress.
    We currently don't have a good way to parameterize
    per sample Dirichilet-Multinomials.
    """
    def __init__(self,
                 table: biom.table.Table,
                 metadata: pd.DataFrame,
                 category_column: str,
                 reference : str = None,
                 num_iter: int = 500,
                 num_warmup: int = 100,
                 chains: int = 4,
                 seed: float = 42,
                 beta_prior: float = 5.0):
        filepath =  os.path.join(os.path.dirname(__file__),
                                 'assets/aldex2.stan')
        warnings.warn('ALDEx2 is not correctly parameterized.')
        super().__init__(table=table,
                         model_path=filepath,
                         num_iter=num_iter,
                         num_warmup=num_warmup,
                         chains=chains,
                         seed=seed)
        cats = metadata[category_column]
        if reference is None:
            reference = cats[0]
        other = list(set(cats) - {reference})[0]
        cats = (cats.values != reference).astype(np.int64) + 1
        param_dict = {
            "depth": table.sum(axis='sample').astype(np.int64),
            "M": cats
        }
        self.add_parameters(param_dict)
        self.specify_model(
            params=["beta"],
            dims={
                "beta": ["group", "feature"],
                "log_lhood": ["tbl_sample", "feature"],
                "y_predict": ["tbl_sample", "feature"]
            },
            coords={
                "group": [reference, other],
                "feature": self.feature_names,
                "tbl_sample": self.sample_names
            },
            include_observed_data=True,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood"
        )
