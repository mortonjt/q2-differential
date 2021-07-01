import biom
import pandas as pd
import birdman
from scipy.stats.mstats import gmean


class DESeq2(birdman.model_base.BaseModel):
    def __init__(self,
                 table: biom.table.Table,
                 metadata: pd.DataFrame,
                 category_column: str,
                 reference: str = None,
                 beta_s : float = 1,
                 alpha_s : float = 1,
                 num_iter: int = 500,
                 num_warmup: int = None,
                 chains: int = 4,
                 seed: float = 42,
                 beta_prior: float = 5.0,
                 cauchy_scale: float = 5.0):
    filepath =  os.path.join(os.path.dirname(__file__),
                                 'assets/deseq2.stan')
    super.__init__(table=table,
                   metadata=metadata,
                   model_path=file_path,
                   num_iter=num_iter,
                   num_warmup=num_warmup,
                   chains=chains
                   seed=seed)
    cats = metadata[category_column]
    if reference is None:
        reference = cats[0]
    cats = (cats.values != reference)

    # compute normalization
    K = table.matrix_data.todense().T + 0.5
    Km = gmean(counts, axis=0)
    slog = np.log(np.median(K / Km, axis=1))

    param_dict = {
        "slog": slog,
        "M": cats,
        "alpha_s": alpha_s,
        "beta_s": beta_s,
    }
    self.add_parameters(param_dict)
    self.specify_model(
        params=["beta_int", "beta_diff", "disp"],
        dims={
            "beta_int": ["feature"],
            "beta_diff": ["feature"],
            "alpha": ["feature"],
            "log_lhood": ["tbl_sample", "feature"],
            "y_predict": ["tbl_sample", "feature"]
        },
        coords={
            "covariate": self.colnames,
            "feature": self.feature_names,
            "tbl_sample": self.sample_names,
            "group": self.groups
        },
        include_observed_data=True,
        posterior_predictive="y_predict",
        log_likelihood="log_lhood"
    )


class ALDEx2(birdman.model_base.BaseModel):
    def __init__(self,
                 table: biom.table.Table,
                 formula: str,
                 metadata: pd.DataFrame,
                 single_feature: bool = False,
                 num_iter: int = 500,
                 num_warmup: int = None,
                 chains: int = 4,
                 seed: float = 42,
                 beta_prior: float = 5.0,
                 cauchy_scale: float = 5.0):
        filepath =  os.path.join(os.path.dirname(__file__),
                                 'assets/aldex2.stan')
        super.__init__(table=table,
                       metadata=metadata,
                       model_path=file_path,
                       num_iter=num_iter,
                       num_warmup=num_warmup,
                       chains=chains
                       seed=seed)
        cats = metadata[category_column]
        if reference is None:
            reference = cats[0]
        cats = (cats.values != reference)
        other = list(set(cats) - {reference})[0]
        param_dict = {
            "depth": table.sum(axis='sample'),
            "M": cats
        }
        self.add_parameters(param_dict)
        self.specify_model(
            params=["beta", "beta_diff", "disp"],
            dims={
                "beta": ["groups", "feature"],
                "beta_diff": ["feature"],
                "alpha": ["feature"],
                "log_lhood": ["tbl_sample", "feature"],
                "y_predict": ["tbl_sample", "feature"]
            },
            coords={
                "groups": [reference, other],
                "feature": self.feature_names,
                "tbl_sample": self.sample_names,
                "group": self.groups
            },
            include_observed_data=True,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood"
        )
