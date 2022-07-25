import os
import biom
import numpy as np
import pandas as pd
import birdman
from scipy.stats.mstats import gmean
from birdman.model_base import TableModel, SingleFeatureModel
from sklearn.preprocessing import LabelEncoder
import warnings


def _normalization_func(table, norm='depth'):
    if norm == 'median_ratios':
        K = table.matrix_data.todense().T + 0.5
        Km = gmean(counts, axis=0)
        slog = np.log(np.median(K / Km, axis=1))
    elif norm == 'depth':
        slog = np.log(table.sum(axis='sample'))
    else:
        raise ValueError('`normalization` must be specified.')
    return slog

def _swap(vec, x, y):
    idx = (vec == x)
    idy = (vec == y)
    new_vec = vec.copy()
    new_vec[idx] = y
    new_vec[idy] = x
    return new_vec
    
class DiseaseSingle(SingleFeatureModel): 
    """A model includes multiple diseases. 
    
    Parameters
    ----------
    table : biom.Table
        Table of counts
    feature_id : str
        Name of feature of interest
    metadata : pd.DataFrame
        Sample metadata file
    ...
    
    """
    def __init__(self,
                 table: biom.Table,
                 feature_id : str,
                 metadata: pd.DataFrame,
                 category_column: str,
                 match_ids_column : str,
                 batch_column : str,
                 reference: str,
                 beta_s : float = 1,
                 alpha_s : float = 1,
                 diff_scale = 1,
                 disp_scale = 0.1,
                 num_iter: int = 500,
                 num_warmup: int = None,
                 normalization: str = 'depth',
                 chains: int = 4,
                 seed: float = 42):
        filepath =  os.path.join(os.path.dirname(__file__),
                                 'assets/disease_single.stan')
        super().__init__(table=table,
                         feature_id=feature_id,
                         model_path=filepath,
                         num_iter=num_iter,
                         num_warmup=num_warmup,
                         chains=chains,
                         seed=seed)
        # pulls down the category information (i.e. health vs different diseases)
        cats = metadata[category_column]
        #LableEncoder values were ranked by letters
        #cats = cats.replace("Healthy", "AAHealthy") 
        disease_encoder = LabelEncoder()
        disease_encoder.fit(cats.values)
        disease_ids = disease_encoder.transform(cats)
        # Swap with reference
        reference_cat = disease_encoder.transform([reference])
        first_cat = disease_encoder.transform([cats[0]])
        disease_ids = _swap(disease_ids, first_cat, reference_cat)
        classes_ = disease_encoder.classes_.copy()
        disease_encoder.classes_ = _swap(classes_, classes_[0], 
                                         classes_[classes_ == reference][0])
        

        disease = disease_encoder.classes_[1:]  # careful here
        #disease = disease_encoder.classes_
         # sequence depth normalization constant
        slog = _normalization_func(table, normalization)
        
        # match ids : convert names to numbers
        case_encoder = LabelEncoder()
        case_ctrl_ids = metadata[match_ids_column].values
        case_encoder.fit(case_ctrl_ids)
        case_ids = case_encoder.transform(case_ctrl_ids)
        
        # batch ids : convert names to numbers (i.e. studies)
        cats = metadata[batch_column]
        batch_encoder = LabelEncoder()
        batch_encoder.fit(cats.values)
        batch_ids = batch_encoder.transform(cats)
        
        C = len(metadata) // 2
        N = len(metadata)
        B = len(np.unique(batch_ids))
        D = len(np.unique(disease_ids)) - 1
        
        control_loc = np.log(1. / len(table.ids(axis='observation')))
        control_scale = 5
        batch_scale = 3
        param_dict = {
            "C" : C,
            "N" : N,
            "B" : B,
            "D" : D,
            "slog": slog, 
            "disease_ids": disease_ids,
            "cc_ids": case_ids + 1,                 # matching ids
            "batch_ids" : batch_ids + 1,            # aka study ids
            "control_loc": control_loc,
            "control_scale": control_scale,
            "batch_scale":batch_scale,
            "diff_scale": diff_scale,
            "disp_scale": disp_scale
        }
        self.add_parameters(param_dict)
        self.specify_model(
            # TODO: specify priors for all parameters
            params=["a0", "a1", "a2", 
                    "diff", "disease_disp"],
            dims={
                "a0": ["feature"],
                "a1": ["feature"],
                "a2": ["feature"],
                "diff": ["feature", "disease_ids"],
                "disp_scale": ["feature", "disease_1p"],
                # TODO: fill out the dimensions for the other parameters
                "log_lhood": ["tbl_sample"],
                "y_predict": ["tbl_sample"]
            },
            coords={
                "groups": [reference, list(disease)],
                "features": [f'log_fold_change'],
                "tbl_samples": self.sample_names
            },
            include_observed_data=True,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood"
        )
  

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
        slog = _normalization_func(table, normalization)
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
                "groups": [reference, other],
                "features": self.feature_names,
                "tbl_samples": self.sample_names
            },
            include_observed_data=True,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood"
        )


class SingleDESeq2(SingleFeatureModel):
    """ A model to mimic DESeq2. """
    def __init__(self,
                 table: biom.table.Table,
                 feature_id : str,
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
                                 'assets/deseq2_single.stan')
        super().__init__(table=table,
                         feature_id=feature_id,
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
        slog = _normalization_func(table, normalization)
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
                "alpha": ["group"],
                "log_lhood": ["tbl_sample"],
                "y_predict": ["tbl_sample"]
            },
            coords={
                "groups": [reference, other],
                "features": [f'log({other} / {reference})'],
                "tbl_samples": self.sample_names
            },
            include_observed_data=True,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood"
        )
