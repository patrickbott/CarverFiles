## This code is required for: Chapter 25
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

DATA_START = datetime(2000,1,1)

class stdevEstimate(dict):
    def list_of_keys(self) -> list:
        return list(self.keys())

    def assets_with_data(self) -> list:
        list_of_assets_with_data = [asset_name for
                                    asset_name, stdev in
                                    self.items()
                                    if not np.isnan(stdev)]

        return list_of_assets_with_data

    def list_in_key_order(self, list_of_assets) -> list:
        return [self[asset_name] for asset_name in list_of_assets]


class genericMatrixEstimate():
    def __init__(self, values: np.array, columns: list):
        if type(values) is pd.DataFrame:
            columns = values.columns
            values = values.values

        self._values = values
        self._columns = columns

    def __repr__(self):
        return str(self.as_df())

    def subset(self, subset_of_asset_names: list):
        as_df = self.as_df()
        subset_df = as_df.loc[subset_of_asset_names, subset_of_asset_names]

        new_correlation = self.from_pd(subset_df)

        return new_correlation

    def add_assets_with_nan_values(self,  new_asset_names: list):
        l1 = self.as_df()
        r1 = pd.DataFrame(
            [[np.nan] * len(new_asset_names)] * len(self.columns),
            columns=new_asset_names,
            index=self.columns,
        )
        top_row = pd.concat([l1, r1], axis=1)
        r2 = pd.DataFrame(
            [[np.nan] * len(new_asset_names)] * len(new_asset_names),
            columns=new_asset_names,
            index=new_asset_names,
        )
        l2 = pd.DataFrame(
            [[np.nan] * len(self.columns)] * len(new_asset_names),
            columns=self.columns,
            index=new_asset_names,
        )
        bottom_row = pd.concat([l2, r2], axis=1)
        both_rows = pd.concat([top_row, bottom_row], axis=0)

        new_cmatrix = correlationEstimate(
            values=both_rows.values, columns=both_rows.columns
        )

        return new_cmatrix


    @classmethod
    def from_pd(cls, matrix_as_pd: pd.DataFrame):
        return cls(matrix_as_pd.values, columns = list(matrix_as_pd.columns))

    def assets_with_data(self) -> list:
        missing = self.assets_with_missing_data()
        return [keyname for keyname in self.columns if keyname not in missing]

    def assets_with_missing_data(self) -> list:
        na_row_count = (~self.as_df().isna()).sum() < 2
        return [keyname for keyname in na_row_count.keys() if na_row_count[keyname]]

    def as_df(self):
        return pd.DataFrame(self.values, columns = self.columns, index = self.columns)

    @property
    def size(self):
        return len(self.columns)

    @property
    def values(self):
        return self._values

    @property
    def columns(self):
        return self._columns



class covarianceEstimate(genericMatrixEstimate):
    pass

class correlationEstimate(genericMatrixEstimate):

    def shrink_to_offdiag(self, offdiag = 0.0, shrinkage_corr: float = 1.0):
        prior_corr = self.boring_corr_matrix(offdiag=offdiag)

        return self.shrink(prior_corr=prior_corr,
                           shrinkage_corr=shrinkage_corr)

    def boring_corr_matrix(self, offdiag: float = 0.99, diag: float = 1.0):

        return create_boring_corr_matrix(
            self.size, offdiag=offdiag, diag=diag, columns=self.columns
        )


    def shrink(self, prior_corr: "correlationEstimate",
               shrinkage_corr: float = 1.0):

        if shrinkage_corr == 1.0:
            return prior_corr

        if shrinkage_corr == 0.0:
            return self

        corr_values = self.values
        prior_corr_values = prior_corr.values

        shrunk_corr = (
            shrinkage_corr * prior_corr_values + (1 - shrinkage_corr) * corr_values
        )

        shrunk_corr = correlationEstimate(shrunk_corr, columns=self.columns)

        return shrunk_corr


def create_boring_corr_matrix(
    size: int, columns: list, offdiag: float = 0.99, diag: float = 1.0
) -> correlationEstimate:

    corr_matrix_values = boring_corr_matrix_values(size, offdiag=offdiag, diag=diag)

    boring_corr_matrix = correlationEstimate(corr_matrix_values, columns=columns)
    boring_corr_matrix.is_boring = True

    return boring_corr_matrix

def boring_corr_matrix_values(
    size: int, offdiag: float = 0.99, diag: float = 1.0
) -> np.array:

    size_index = range(size)

    def _od(i, j, offdiag, diag):
        if i == j:
            return diag
        else:
            return offdiag

    corr_matrix_values_as_list = [
        [_od(i, j, offdiag, diag) for i in size_index] for j in size_index
    ]
    corr_matrix_values = np.array(corr_matrix_values_as_list)

    return corr_matrix_values


@dataclass
class covarianceList:
    cov_list: list
    fit_dates: list

    def most_recent_covariance_before_date(
        self, relevant_date: datetime
    ) -> covarianceEstimate:

        index_of_date = get_max_index_before_datetime(self.fit_dates,
                                                      relevant_date)
        if index_of_date is None:
            ## slightly forward looking but likely to be all Nan anyway
            index_of_date = 0

        return self.cov_list[index_of_date]




def calculate_covariance_matrices(adjusted_prices_dict: dict,
                                   std_dev_dict: dict,
                                   current_prices_dict: dict) \
                                -> covarianceList:

    weekly_df_of_percentage_returns = get_weekly_df_of_percentage_returns(
        adjusted_prices_dict=adjusted_prices_dict,
        current_prices_dict=current_prices_dict
    )

    exp_correlations = calculate_exponential_correlations(
        weekly_df_of_percentage_returns
    )

    weekly_index = weekly_df_of_percentage_returns.index
    print("Calculating covariance - can take a while")
    list_of_covariance = [
        calculate_covariance_matrix_at_date(
                                    relevant_date,
                                    std_dev_dict = std_dev_dict,
                                    exp_correlations = exp_correlations
                        )
                for relevant_date in weekly_index]

    return covarianceList(cov_list=list_of_covariance,
                          fit_dates=weekly_index)

def get_weekly_df_of_percentage_returns(adjusted_prices_dict: dict,
                                   current_prices_dict: dict) -> pd.DataFrame:

    weekly_common_index = get_common_weekly_index(adjusted_prices_dict)
    list_of_instruments = list(adjusted_prices_dict.keys())
    dict_of_perc_returns = dict(
        [
            (
            instrument_code,
            calculate_weekly_percentage_returns(
                adjusted_price=adjusted_prices_dict[instrument_code],
                current_price=current_prices_dict[instrument_code],
                weekly_common_index = weekly_common_index)
            )
            for instrument_code in list_of_instruments
        ]
    )

    df_of_percentage_returns = pd.concat(dict_of_perc_returns, axis=1)
    df_of_percentage_returns.columns = list_of_instruments

    return df_of_percentage_returns

def calculate_weekly_percentage_returns(adjusted_price: pd.Series,
                                 current_price: pd.Series,
                                weekly_common_index: list) -> pd.Series:

    weekly_adj_prices = adjusted_price.reindex(weekly_common_index, method="ffill")
    weekly_current_prices = current_price.reindex(weekly_common_index, method="ffill")
    daily_price_changes = weekly_adj_prices.diff()
    percentage_changes = daily_price_changes / weekly_current_prices.shift(1)

    return percentage_changes

def calculate_exponential_correlations(
        weekly_df_of_percentage_returns: pd.DataFrame) -> pd.DataFrame:

    raw_correlations = weekly_df_of_percentage_returns.ewm(
        span=25, min_periods=3, ignore_na=True
    ).corr(pairwise=True, ignore_na=True)

    return raw_correlations

def calculate_covariance_matrix_at_date(
                                    relevant_date: datetime,
                                   std_dev_dict: dict,
                                   exp_correlations: pd.DataFrame) \
        -> covarianceEstimate:

    columns = list(std_dev_dict.keys())
    correlation_estimate = get_correlation_estimate_at_date(relevant_date,
                                                            columns=columns,
                                                            exp_correlations=exp_correlations)

    ## We have to do this because we sometimes get slightly non PSD matrices
    ## It also slows the turnover of the DO strategy down slightly

    correlation_estimate = correlation_estimate.shrink_to_offdiag(offdiag=0,
                                                                  shrinkage_corr=0.5)

    stdev_estimate = stdevEstimate(get_values_for_date_as_dict(relevant_date, std_dev_dict))

    covariance_estimate = calculate_covariance_given_correlation_and_stdev(
        correlation_estimate = correlation_estimate,
        stdev_estimate = stdev_estimate
    )

    return covariance_estimate

def get_correlation_estimate_at_date(relevant_date: datetime,
                                     columns: list,
                                   exp_correlations: pd.DataFrame) -> correlationEstimate:

    size_of_matrix = len(columns)
    corr_matrix_values = (
        exp_correlations[exp_correlations.index.get_level_values(0) < relevant_date]
        .tail(size_of_matrix)
        .values
    )

    if corr_matrix_values.shape[0]==0:
        ## empty
        corr_matrix_values = np.array([[np.nan]*len(columns)]*len(columns))

    return correlationEstimate(values=corr_matrix_values, columns=columns)


def get_values_for_date_as_dict(relevant_date: datetime,
                                dict_with_values: dict) -> dict:
    values_as_dict = dict(
        [
            (key_name,
             get_row_of_series_before_date(ts_series, relevant_date=relevant_date)
             )
            for key_name, ts_series in dict_with_values.items()
        ]
    )

    return values_as_dict

def get_row_of_series_before_date(
        series: pd.Series, relevant_date: datetime
):
    index_point = get_max_index_before_datetime(series.index,
                                                relevant_date)
    if index_point is None:
        return np.nan

    data_at_date = series.values[index_point]

    return data_at_date

def get_max_index_before_datetime(index, date_point):
    matching_index_size = index[index < date_point].size

    if matching_index_size == 0:
        return None
    else:
        return matching_index_size - 1

def get_common_index(some_dict: dict) -> list:
    all_stuff = pd.concat(some_dict, axis=1)
    all_stuff = all_stuff[DATA_START:]
    return all_stuff.index

def get_common_weekly_index(some_dict: dict) -> list:
    all_stuff = pd.concat(some_dict, axis=1)
    all_stuff = all_stuff.resample("7D").last()
    all_stuff = all_stuff[DATA_START:]
    return all_stuff.index

def calculate_covariance_given_correlation_and_stdev(
        correlation_estimate: correlationEstimate,
        stdev_estimate: stdevEstimate
    ) -> covarianceEstimate:

    all_assets = set(list(correlation_estimate.columns) + stdev_estimate.list_of_keys())
    list_of_assets_with_data = list(
        set(correlation_estimate.assets_with_data()).intersection(
            set(stdev_estimate.assets_with_data())
        )
    )
    assets_without_data = list(all_assets.difference(list_of_assets_with_data))

    aligned_stdev_list = stdev_estimate.list_in_key_order(list_of_assets_with_data)
    aligned_corr_list = correlation_estimate.subset(list_of_assets_with_data)
    covariance_as_np_array = \
        sigma_from_corr_and_std(aligned_stdev_list, aligned_corr_list.values)

    covariance_assets_with_data = covarianceEstimate(
        covariance_as_np_array, columns=list_of_assets_with_data)

    covariance = covariance_assets_with_data.add_assets_with_nan_values(assets_without_data)

    return covariance


def sigma_from_corr_and_std(stdev_list: list, corrmatrix: np.array):
    sigma = np.diag(stdev_list).dot(corrmatrix).dot(np.diag(stdev_list))
    return sigma


