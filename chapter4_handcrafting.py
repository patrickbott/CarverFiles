"""
This is the provided example python code for the handcrafting method for instruments outlined Chapter four of the book:
 "Advanced Futures Trading Strategies", by Robert Carver
 https://www.systematicmoney.org/advanced-futures

This code is copyright, Robert Carver 2022.
Shared under https://www.gnu.org/licenses/gpl-3.0.en.html
You may copy, modify, and share this code as long as this header is retained, and you disclose that it has been edited.
This code comes with no warranty, is not guaranteed to be accurate, and the author is not responsible for any losses that may result from it’s use.

Results may not match the book exactly as different data may be used
Results may be different from the corresponding spreadsheet as methods may be slightly different

"""
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy as sch

from chapter1 import pd_readcsv

PRINT_TRACE = False

class correlationEstimate(object):
    def __init__(self, values: pd.DataFrame):
        columns = values.columns
        values = values.values

        self._values = values
        self._columns = columns

    def __repr__(self):
        return str(self.as_pd())

    def __len__(self):
        return len(self.columns)

    def as_pd(self) -> pd.DataFrame:
        values = self.values
        columns = self.columns

        return pd.DataFrame(values, index=columns, columns=columns)

    @property
    def values(self) -> np.array:
        return self._values

    @property
    def columns(self) -> list:
        return self._columns

    @property
    def size(self) -> int:
        return len(self.columns)

    def subset(self, subset_of_asset_names: list):
        as_pd = self.as_pd()
        subset_pd = as_pd.loc[subset_of_asset_names, subset_of_asset_names]

        new_correlation = correlationEstimate(subset_pd)

        return new_correlation


def cluster_correlation_matrix(corr_matrix: correlationEstimate,
                               cluster_size: int = 2):

    clusters = get_list_of_clusters_for_correlation_matrix(corr_matrix,
                                                                          cluster_size=cluster_size)
    clusters_as_names = from_cluster_index_to_asset_names(clusters, corr_matrix)
    if PRINT_TRACE:
        print("Cluster split: %s" % str(clusters_as_names))

    return clusters_as_names


def get_list_of_clusters_for_correlation_matrix(corr_matrix: np.array,
                                             cluster_size: int = 2) -> list:
    corr_as_np = corr_matrix.values
    try:
        clusters = get_list_of_clusters_for_correlation_matrix_as_np(
            corr_as_np,
            cluster_size=cluster_size
        )
    except:
        clusters = arbitrary_split_of_correlation_matrix(
            corr_as_np,
            cluster_size=cluster_size
        )

    return clusters

def get_list_of_clusters_for_correlation_matrix_as_np(corr_as_np: np.array,
                                             cluster_size: int = 2) -> list:
    d = sch.distance.pdist(corr_as_np)
    L = sch.linkage(d, method="complete")

    cutoff = cutoff_distance_to_guarantee_N_clusters(corr_as_np, L=L,
                                                     cluster_size = cluster_size)
    ind = sch.fcluster(L, cutoff, "distance")
    ind = list(ind)

    if max(ind) > cluster_size:
        raise Exception("Couldn't cluster into %d clusters" % cluster_size)

    return ind


def cutoff_distance_to_guarantee_N_clusters(corr_as_np: np.array, L: np.array,
                                            cluster_size: int = 2):
    #assert cluster_size==2

    N = len(corr_as_np)
    return L[N - cluster_size][2] - 0.000001


def arbitrary_split_of_correlation_matrix(corr_matrix: np.array,
                                          cluster_size: int = 2) -> list:
    # split correlation of 3 or more assets
    count_assets = len(corr_matrix)
    return arbitrary_split_for_asset_length(count_assets, cluster_size=cluster_size)


def arbitrary_split_for_asset_length(count_assets: int,
                                     cluster_size: int = 2) -> list:

    return [(x % cluster_size) + 1 for x in range(count_assets)]


def from_cluster_index_to_asset_names(
    clusters: list, corr_matrix: correlationEstimate
) -> list:

    all_clusters = list(set(clusters))
    asset_names = corr_matrix.columns
    list_of_asset_clusters = [
        get_asset_names_for_cluster_index(cluster_id, clusters, asset_names)
        for cluster_id in all_clusters
    ]

    return list_of_asset_clusters


def get_asset_names_for_cluster_index(
    cluster_id: int, clusters: list, asset_names: list
):

    list_of_assets = [
        asset for asset, cluster in zip(asset_names, clusters) if cluster == cluster_id
    ]

    return list_of_assets


class portfolioWeights(dict):
    @property
    def weights(self):
        return list(self.values())

    @property
    def assets(self):
        return list(self.keys())

    def multiply_by_float(self, multiplier: float):
        list_of_assets = self.assets
        list_of_weights = [self[asset] for asset in list_of_assets]
        list_of_weights_multiplied = [weight * multiplier for weight in list_of_weights]
        return portfolioWeights.from_weights_and_keys(
            list_of_weights=list_of_weights_multiplied,
            list_of_keys=list_of_assets
        )

    @classmethod
    def from_list_of_subportfolios(portfolioWeights, list_of_portfolio_weights):
        list_of_unique_asset_names = list(
            set(
                flatten_list(
                    [
                        subportfolio.assets
                        for subportfolio in list_of_portfolio_weights
                    ]
                )
            )
        )

        portfolio_weights = portfolioWeights.allzeros(list_of_unique_asset_names)

        for subportfolio_weights in list_of_portfolio_weights:
            for asset_name in subportfolio_weights.assets:
                portfolio_weights[asset_name] = (
                    portfolio_weights[asset_name] + subportfolio_weights[asset_name]
                )

        return portfolio_weights

    @classmethod
    def allzeros(portfolioWeights, list_of_keys: list):
        return portfolioWeights.all_one_value(list_of_keys, value=0.0)

    @classmethod
    def all_one_value(portfolioWeights, list_of_keys: list, value=0.0):
        return portfolioWeights.from_weights_and_keys(
            list_of_weights=[value] * len(list_of_keys), list_of_keys=list_of_keys
        )

    @classmethod
    def from_weights_and_keys(
        portfolioWeights, list_of_weights: list, list_of_keys: list
    ):
        assert len(list_of_keys) == len(list_of_weights)
        pweights_as_list = [
            (key, weight) for key, weight in zip(list_of_keys, list_of_weights)
        ]

        return portfolioWeights(pweights_as_list)

def flatten_list(some_list):
    flattened = [item for sublist in some_list for item in sublist]

    return flattened


def one_over_n_weights_given_asset_names(list_of_asset_names: list) -> portfolioWeights:
    weight = 1.0 / len(list_of_asset_names)
    return portfolioWeights(
        [(asset_name, weight) for asset_name in list_of_asset_names]
    )


class handcraftPortfolio(object):
    def __init__(self, correlation: correlationEstimate):

        self._correlation = correlation

    @property
    def correlation(self) -> correlationEstimate:
        return self._correlation

    @property
    def size(self) -> int:
        return len(self.correlation)

    @property
    def asset_names(self) -> list:
        return list(self.correlation.columns)

    def weights(self) -> portfolioWeights:
        if self.size <= 2:
            # don't cluster one or two assets
            raw_weights = self.risk_weights_this_portfolio()
        else:
            raw_weights = self.aggregated_risk_weights()

        return raw_weights

    def risk_weights_this_portfolio(self) -> portfolioWeights:

        asset_names = self.asset_names
        raw_weights = one_over_n_weights_given_asset_names(asset_names)

        return raw_weights

    def aggregated_risk_weights(self):
        sub_portfolios = create_sub_portfolios_from_portfolio(self)
        aggregate_risk_weights = aggregate_risk_weights_over_sub_portfolios(
            sub_portfolios
        )

        return aggregate_risk_weights

    def subset(self, subset_of_asset_names: list):
        return handcraftPortfolio(self.correlation.subset(subset_of_asset_names))



## SUB PORTFOLIOS


def create_sub_portfolios_from_portfolio(handcraft_portfolio: handcraftPortfolio):

    clusters_as_names = \
        cluster_correlation_matrix(handcraft_portfolio.correlation)

    sub_portfolios = create_sub_portfolios_given_clusters(
        clusters_as_names, handcraft_portfolio
    )

    return sub_portfolios


def create_sub_portfolios_given_clusters(
    clusters_as_names: list, handcraft_portfolio: handcraftPortfolio
) -> list:
    list_of_sub_portfolios = [
        handcraft_portfolio.subset(subset_of_asset_names)
        for subset_of_asset_names in clusters_as_names
    ]

    return list_of_sub_portfolios


def aggregate_risk_weights_over_sub_portfolios(
    sub_portfolios: list,
) -> portfolioWeights:
    # sub portfolios guaranteed to be 2 long
    # We allocate half to each
    asset_count = len(sub_portfolios)
    #assert asset_count == 2
    weights_for_each_subportfolio = [1.0/asset_count]*asset_count

    risk_weights_by_portfolio = [
        sub_portfolio.weights() for sub_portfolio in
        sub_portfolios
    ]

    multiplied_risk_weights_by_portfolio = [
        sub_portfolio_weights.multiply_by_float(weight_for_subportfolio) for
        weight_for_subportfolio, sub_portfolio_weights in
        zip(weights_for_each_subportfolio, risk_weights_by_portfolio)
    ]

    aggregate_weights = portfolioWeights.from_list_of_subportfolios(
        multiplied_risk_weights_by_portfolio
    )

    return aggregate_weights



if __name__ == '__main__':
    ## Get the file from: https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/example_instrument_returns.csv
    all_returns = pd_readcsv('example_instrument_returns.csv')
    #
    ## NOTE: Strictly speaking we should use instrument sub-strategy returns here, however
    ##       using the returns of underlying instruments won't make much difference
    ##  In the .csv we are indeed using instrument sub-strategy returns, eg the output from perc_returns_to_df(perc_returns_dict) in chapter4.py

    all_returns = pd_readcsv('example_instrument_returns.csv')
    corr_matrix = correlationEstimate(all_returns.corr())

    ## The weights here are produced by recursive clustering of the correlation matrix
    ##   into two clusters.

    handcraft_portfolio = handcraftPortfolio(corr_matrix)
    PRINT_TRACE = True
    print(handcraft_portfolio.weights())

"""
Safe rates/everything else
Cluster split: [['BUND', 'US10', 'EDOLLAR'], ['SOYBEAN', 'MILKWET', 'CORN', 'BTP', 'CAC', 'DAX', 'NASDAQ', 'NOK', 'GOLD', 'SILVER', 'COPPER', 'ETHANOL', 'VIX']]

US rates/ German rates
Cluster split: [['US10', 'EDOLLAR'], ['BUND']]

BTP+equity / commodities
Cluster split: [['BTP', 'CAC', 'DAX', 'NASDAQ', 'VIX'], ['SOYBEAN', 'MILKWET', 'CORN', 'NOK', 'GOLD', 'SILVER', 'COPPER', 'ETHANOL']]

Equity / BTP
Cluster split: [['CAC', 'DAX', 'NASDAQ', 'VIX'], ['BTP']]

French+US equity / German equity (not clear why this split)
Cluster split: [['CAC', 'NASDAQ', 'VIX'], ['DAX']]

Equity / vol
Cluster split: [['CAC', 'NASDAQ'], ['VIX']]

NOK+metal / Ags+Energy
Cluster split: [['NOK', 'GOLD', 'SILVER', 'COPPER'], ['SOYBEAN', 'MILKWET', 'CORN', 'ETHANOL']]

Precious metal / NOK + copper
Cluster split: [['GOLD', 'SILVER'], ['NOK', 'COPPER']]

Dry commodities / Wet commodities
Cluster split: [['SOYBEAN', 'CORN'], ['MILKWET', 'ETHANOL']]

Weights:
{'CAC': 0.015625, 'US10': 0.125, 'NASDAQ': 0.015625, 'MILKWET': 0.03125, 'SOYBEAN': 0.03125, 'DAX': 0.0625, 'EDOLLAR': 0.125, 'VIX': 0.03125, 'NOK': 0.03125, 'SILVER': 0.03125, 'COPPER': 0.03125, 'ETHANOL': 0.03125, 'CORN': 0.03125, 'GOLD': 0.03125, 'BUND': 0.25, 'BTP': 0.125}

"""