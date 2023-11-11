"""
This is the provided example python code for the static instrument selection outlined Chapter four of the book:
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
from chapter1 import pd_readcsv
from chapter4 import minimum_capital_for_sub_strategy
from chapter4_handcrafting import correlationEstimate, portfolioWeights, handcraftPortfolio

def select_first_static_instrument(instrument_config: pd.DataFrame,
                                   approx_number_of_instruments: int,
                                   approx_IDM: float,
                                   capital: float,
                                   risk_target: float,
                                   position_turnover: float):

    approx_initial_weight = 1.0/ approx_number_of_instruments
    instrument_list = list(instrument_config.index)
    instruments_okay_for_minimum_capital = [instrument_code
                         for instrument_code in instrument_list
                         if minimum_capital_okay_for_instrument(
                                instrument_code=instrument_code,
                                instrument_config=instrument_config,
                                capital=capital,
                                weight=approx_initial_weight,
                                idm = approx_IDM,
                                risk_target=risk_target
                                )]

    cheapest_instrument = lowest_risk_adjusted_cost_given_instrument_list(
        instruments_okay_for_minimum_capital,
        instrument_config=instrument_config,
        position_turnover=position_turnover
    )

    return cheapest_instrument

def minimum_capital_okay_for_instrument(instrument_code: str,
                                         instrument_config: pd.DataFrame,
                                         idm: float,
                                         weight: float,
                                         risk_target: float,
                                         capital: float) -> bool:

    config_for_instrument = instrument_config.loc[instrument_code]
    minimum_capital = minimum_capital_for_sub_strategy(
        fx = config_for_instrument.fx_rate,
        idm = idm,
        weight=weight,
        instrument_risk_ann_perc=config_for_instrument.ann_std,
        price=config_for_instrument.price,
        multiplier=config_for_instrument.multiplier,
        risk_target=risk_target
    )

    return minimum_capital<=capital

def lowest_risk_adjusted_cost_given_instrument_list(
        instrument_list: list,
        instrument_config: pd.DataFrame,
        position_turnover: float
        ) -> str:

    list_of_risk_adjusted_cost_by_instrument = [
        risk_adjusted_cost_for_instrument(instrument_code,
                                          instrument_config = instrument_config,
                                          position_turnover = position_turnover)
        for instrument_code in instrument_list
    ]
    index_min = get_min_index(list_of_risk_adjusted_cost_by_instrument)
    return instrument_list[index_min]

def get_min_index(x: list) -> int:
    index_min = get_func_index(x, min)
    return index_min

def get_max_index(x: list) -> int:
    index_max = get_func_index(x, max)
    return index_max

def get_func_index(x: list, func) -> int:
    index_min = func(range(len(x)),
                    key=x.__getitem__)

    return index_min


def risk_adjusted_cost_for_instrument(instrument_code: str,
                                      instrument_config: pd.DataFrame,
                                      position_turnover: float) -> float:

    config_for_instrument = instrument_config.loc[instrument_code]
    SR_cost_per_trade = config_for_instrument.SR_cost
    rolls_per_year = config_for_instrument.rolls_per_year

    return SR_cost_per_trade * (rolls_per_year + position_turnover)

def calculate_SR_for_selected_instruments(selected_instruments: list,
                                          pre_cost_SR: float,
                                          instrument_config: pd.DataFrame,
                                          position_turnover: float,
                                          correlation_matrix: correlationEstimate,
                                        capital: float,
                                          risk_target: float
                                          ) -> float:

    ## Returns a large negative number if minimum capital requirements not met

    portfolio_weights = calculate_portfolio_weights(selected_instruments,
                                                    correlation_matrix=correlation_matrix)

    min_capital_okay = check_minimum_capital_ok(
        portfolio_weights=portfolio_weights,
        instrument_config=instrument_config,
        correlation_matrix=correlation_matrix,
        risk_target=risk_target,
        capital=capital
    )

    if not min_capital_okay:
        return -999999999999

    portfolio_SR = calculate_SR_of_portfolio(portfolio_weights,
                                             pre_cost_SR=pre_cost_SR,
                                             correlation_matrix=correlation_matrix,
                                             position_turnover=position_turnover,
                                             instrument_config=instrument_config)

    return portfolio_SR


def calculate_portfolio_weights(selected_instruments: list,
                                correlation_matrix: correlationEstimate) -> portfolioWeights:


    if len(selected_instruments)==1:
        return portfolioWeights.from_weights_and_keys(list_of_weights=[1.0],
                                                      list_of_keys=selected_instruments)

    subset_matrix = correlation_matrix.subset(selected_instruments)
    handcraft_portfolio = handcraftPortfolio(subset_matrix)

    return handcraft_portfolio.weights()

def check_minimum_capital_ok(
        portfolio_weights: portfolioWeights,
        correlation_matrix: correlationEstimate,
        risk_target: float,
        instrument_config: pd.DataFrame,
        capital: float
        ) -> bool:

    idm = calculate_idm(portfolio_weights,
                        correlation_matrix=correlation_matrix)

    list_of_instruments = portfolio_weights.assets

    for instrument_code in list_of_instruments:
        weight = portfolio_weights[instrument_code]
        okay_for_instrument = minimum_capital_okay_for_instrument(instrument_code=instrument_code,
                                            instrument_config=instrument_config,
                                            capital=capital,
                                            risk_target=risk_target,
                                            idm = idm,
                                            weight = weight)
        if not okay_for_instrument:
            return False

    return True


def calculate_idm(portfolio_weights: portfolioWeights,
                  correlation_matrix: correlationEstimate) -> float:

    if len(portfolio_weights.assets)==1:
        return 1.0

    aligned_correlation_matrix = correlation_matrix.subset(portfolio_weights.assets)
    return div_multiplier_from_np(np.array(portfolio_weights.weights),
                                  aligned_correlation_matrix.values)


def div_multiplier_from_np(weights_np: np.array,
                   corr_np: np.array
                   ) -> float:

    variance = weights_np.dot(corr_np).dot(weights_np)
    risk = variance ** 0.5

    return 1.0 / risk


def calculate_SR_of_portfolio(portfolio_weights: portfolioWeights,
                    pre_cost_SR: float,
                    instrument_config: pd.DataFrame,
                    position_turnover: float,
                    correlation_matrix: correlationEstimate
    ) -> float:

    expected_mean = calculate_expected_mean_for_portfolio(
        portfolio_weights=portfolio_weights,
        pre_cost_SR=pre_cost_SR,
        instrument_config=instrument_config,
        position_turnover=position_turnover
    )
    expected_std = calculate_expected_std_for_portfolio(
        portfolio_weights=portfolio_weights,
        correlation_matrix=correlation_matrix
    )

    return expected_mean / expected_std

def calculate_expected_mean_for_portfolio(
                                          portfolio_weights: portfolioWeights,
                                          pre_cost_SR: float,
                                          instrument_config: pd.DataFrame,
                                          position_turnover: float
                                          ) -> float:
    instrument_means = [
        calculate_expected_mean_for_instrument_in_portfolio(instrument_code,
                                                            portfolio_weights=portfolio_weights,
                                                            pre_cost_SR=pre_cost_SR,
                                                            instrument_config=instrument_config,
                                                            position_turnover=position_turnover)
        for instrument_code in portfolio_weights.assets
    ]

    return sum(instrument_means)

def calculate_expected_mean_for_instrument_in_portfolio(instrument_code: str,
                                                        portfolio_weights: portfolioWeights,
                                                        pre_cost_SR: float,
                                                        instrument_config: pd.DataFrame,
                                                        position_turnover: float
                                                        ):
    weight = portfolio_weights[instrument_code]
    costs_SR_units = risk_adjusted_cost_for_instrument(instrument_code=instrument_code,
                                                       instrument_config=instrument_config,
                                                       position_turnover=position_turnover)
    SR_for_instrument = pre_cost_SR - costs_SR_units

    return weight *SR_for_instrument

def calculate_expected_std_for_portfolio(portfolio_weights: portfolioWeights,
                                         correlation_matrix: correlationEstimate) -> float:

    subset_aligned_correlation = correlation_matrix.subset(portfolio_weights.assets)

    return variance_for_numpy(weights = np.array(portfolio_weights.weights),
                              sigma = subset_aligned_correlation.values)

def variance_for_numpy(weights: np.array, sigma: np.array) -> float:
    # returns the variance (NOT standard deviation) given weights and sigma
    return weights.dot(sigma).dot(weights.transpose())


def choose_next_instrument(selected_instruments: list,
                          pre_cost_SR: float,
                           capital: float,
                           risk_target: float,
                          instrument_config: pd.DataFrame,
                          position_turnover: float,
                          correlation_matrix: correlationEstimate) -> str:

    remaining_instruments = get_remaining_instruments(selected_instruments,
                                                      instrument_config=instrument_config)

    SR_by_instrument = [
        calculate_SR_for_selected_instruments(selected_instruments+[instrument_code],
                                              correlation_matrix=correlation_matrix,
                                              capital=capital,
                                              pre_cost_SR=pre_cost_SR,
                                              instrument_config=instrument_config,
                                              risk_target=risk_target,
                                              position_turnover=position_turnover)
        for instrument_code in remaining_instruments
    ]

    index_of_max_SR = get_max_index(SR_by_instrument)

    return remaining_instruments[index_of_max_SR]

def get_remaining_instruments(selected_instruments: list,
                          instrument_config: pd.DataFrame) -> list:

    all_instruments = list(instrument_config.index)
    remaining = set(all_instruments).difference(set(selected_instruments))

    return list(remaining)

if __name__ == '__main__':
    ## Get the file from: https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/example_instrument_returns.csv
    all_returns = pd_readcsv('example_instrument_returns.csv')
    #
    # Instruments should be pre-selected to pass liquidity and cost requirements
    #
    ## NOTE: Strictly speaking we should use instrument sub-strategy returns here, however
    ##       using the returns of underlying instruments won't make much difference
    ##  In the .csv we are indeed using instrument sub-strategy returns, eg the output from perc_returns_to_df(perc_returns_dict) in chapter4.py

    all_returns = pd_readcsv('example_instrument_returns.csv')
    correlation_matrix = correlationEstimate(all_returns.corr())

    ## Need capital and tau for min. capital

    ## Need a figure for turnover (assumed to be the same for each instrument)

    ## As well as the correlations, for each instrument we require some metadata:
    # Get the file from here:https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/instrument_config.csv
    instrument_config = pd.read_csv('instrument_config.csv', index_col='instrument')

    capital = 50000000
    risk_target = 0.2
    approx_number_of_instruments =5
    approx_IDM = 2.5
    position_turnover = 5 ## turnover without rolls
    pre_cost_SR = .4 ## per instrument

    selected_instruments = []
    first_instrument = select_first_static_instrument(instrument_config=instrument_config,
                                                      position_turnover=position_turnover,
                                                      capital=capital,
                                                      risk_target=risk_target,
                                                      approx_IDM=approx_IDM,
                                                      approx_number_of_instruments=approx_number_of_instruments)
    selected_instruments.append(first_instrument)

    current_SR = calculate_SR_for_selected_instruments(selected_instruments,
                                                       correlation_matrix=correlation_matrix,
                                                       pre_cost_SR=pre_cost_SR,
                                                       instrument_config=instrument_config,
                                                       position_turnover=position_turnover,
                                                       capital=capital,
                                                       risk_target=risk_target)

    max_SR_achieved = current_SR

    while current_SR>(max_SR_achieved*.9):
        print("%s SR: %.2f" % (str(selected_instruments), current_SR))
        next_instrument = choose_next_instrument(selected_instruments,
                                                correlation_matrix=correlation_matrix,
                                                pre_cost_SR=pre_cost_SR,
                                                instrument_config=instrument_config,
                                                position_turnover=position_turnover,
                                                 capital=capital,
                                                 risk_target=risk_target)
        selected_instruments.append(next_instrument)
        current_SR = calculate_SR_for_selected_instruments(selected_instruments,
                                                           correlation_matrix=correlation_matrix,
                                                           pre_cost_SR=pre_cost_SR,
                                                           instrument_config=instrument_config,
                                                           position_turnover=position_turnover,
                                                           capital=capital,
                                                           risk_target=risk_target)
        if current_SR>max_SR_achieved:
            max_SR_achieved = current_SR

