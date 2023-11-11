"""
This is the provided example python code for Chapter three of the book:
 "Advanced Futures Trading Strategies", by Robert Carver
 https://www.systematicmoney.org/advanced-futures

This code is copyright, Robert Carver 2022.
Shared under https://www.gnu.org/licenses/gpl-3.0.en.html
You may copy, modify, and share this code as long as this header is retained, and you disclose that it has been edited.
This code comes with no warranty, is not guaranteed to be accurate, and the author is not responsible for any losses that may result from it’s use.

Results may not match the book exactly as different data may be used
Results may be different from the corresponding spreadsheet as methods may be slightly different

"""

## Next two lines are optional depending on your IDE
import matplotlib

matplotlib.use("TkAgg")

from copy import copy

import pandas as pd

from chapter1 import (
    pd_readcsv,
    BUSINESS_DAYS_IN_YEAR,
    calculate_perc_returns,
    calculate_stats,
    MONTH,
)
from chapter2 import calculate_minimum_capital


def calculate_variable_standard_deviation_for_risk_targeting(
    adjusted_price: pd.Series,
    current_price: pd.Series,
    use_perc_returns: bool = True,
    annualise_stdev: bool = True,
) -> pd.Series:

    if use_perc_returns:
        daily_returns = calculate_percentage_returns(
            adjusted_price=adjusted_price, current_price=current_price
        )
    else:
        daily_returns = calculate_daily_returns(adjusted_price=adjusted_price)

    ## Can do the whole series or recent history
    daily_exp_std_dev = daily_returns.ewm(span=32).std()

    if annualise_stdev:
        annualisation_factor = BUSINESS_DAYS_IN_YEAR ** 0.5
    else:
        ## leave at daily
        annualisation_factor = 1

    annualised_std_dev = daily_exp_std_dev * annualisation_factor

    ## Weight with ten year vol
    ten_year_vol = annualised_std_dev.rolling(
        BUSINESS_DAYS_IN_YEAR * 10, min_periods=1
    ).mean()
    weighted_vol = 0.3 * ten_year_vol + 0.7 * annualised_std_dev

    return weighted_vol


def calculate_percentage_returns(
    adjusted_price: pd.Series, current_price: pd.Series
) -> pd.Series:

    daily_price_changes = calculate_daily_returns(adjusted_price)
    percentage_changes = daily_price_changes / current_price.shift(1)

    return percentage_changes


def calculate_daily_returns(adjusted_price: pd.Series) -> pd.Series:

    return adjusted_price.diff()


class standardDeviation(pd.Series):
    ## class that can be eithier % or price based standard deviation estimate
    def __init__(
        self,
        adjusted_price: pd.Series,
        current_price: pd.Series,
        use_perc_returns: bool = True,
        annualise_stdev: bool = True,
    ):

        stdev = calculate_variable_standard_deviation_for_risk_targeting(
            adjusted_price=adjusted_price,
            current_price=current_price,
            annualise_stdev=annualise_stdev,
            use_perc_returns=use_perc_returns,
        )
        super().__init__(stdev)

        self._use_perc_returns = use_perc_returns
        self._annualised = annualise_stdev
        self._current_price = current_price

    def daily_risk_price_terms(self):
        stdev = copy(self)
        if self.annualised:
            stdev = stdev / (BUSINESS_DAYS_IN_YEAR ** 0.5)

        if self.use_perc_returns:
            stdev = stdev * self.current_price

        return stdev

    def annual_risk_price_terms(self):
        stdev = copy(self)
        if not self.annualised:
            # daily
            stdev = stdev * (BUSINESS_DAYS_IN_YEAR ** 0.5)

        if self.use_perc_returns:
            stdev = stdev * self.current_price

        return stdev

    @property
    def annualised(self) -> bool:
        return self._annualised

    @property
    def use_perc_returns(self) -> bool:
        return self._use_perc_returns

    @property
    def current_price(self) -> pd.Series:
        return self._current_price


def calculate_position_series_given_variable_risk(
    capital: float,
    risk_target_tau: float,
    fx: pd.Series,
    multiplier: float,
    instrument_risk: standardDeviation,
) -> pd.Series:

    # N = (Capital × τ) ÷ (Multiplier × Price × FX × σ %)
    ## resolves to N = (Capital × τ) ÷ (Multiplier × FX × daily stdev price terms × 16)
    ## for simplicity we use the daily risk in price terms, even if we calculated annualised % returns
    daily_risk_price_terms = instrument_risk.daily_risk_price_terms()

    return (
        capital
        * risk_target_tau
        / (multiplier * fx * daily_risk_price_terms * (BUSINESS_DAYS_IN_YEAR ** 0.5))
    )


def calculate_turnover(position, average_position):
    daily_trades = position.diff()
    as_proportion_of_average = daily_trades.abs() / average_position.shift(1)
    average_daily = as_proportion_of_average.mean()
    annualised_turnover = average_daily * BUSINESS_DAYS_IN_YEAR

    return annualised_turnover


if __name__ == "__main__":
    ## Get the file from https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/sp500.csv
    data = pd_readcsv("sp500.csv")
    data = data.dropna()

    adjusted_price = data.adjusted
    current_price = data.underlying
    multiplier = 5
    risk_target_tau = 0.2
    fx_series = pd.Series(1, index=data.index)  ## FX rate, 1 for USD / USD

    capital = 100000  ## applies only to strategy 1

    ## eithier use annual # % returns, or daily price differences to calculate
    instrument_risk = standardDeviation(
        adjusted_price=adjusted_price,
        current_price=current_price,
        use_perc_returns=True,
        annualise_stdev=True,
    )

    ## or
    """
    instrument_risk = standardDeviation(adjusted_price=adjusted_price,
                                                current_price=current_price,
                                                 use_perc_returns=False,
                                                 annualise_stdev=False)
    """

    position_contracts_held = calculate_position_series_given_variable_risk(
        capital=capital,
        fx=fx_series,
        instrument_risk=instrument_risk,
        risk_target_tau=risk_target_tau,
        multiplier=multiplier,
    )

    perc_return = calculate_perc_returns(
        position_contracts_held=position_contracts_held,
        adjusted_price=adjusted_price,
        fx_series=fx_series,
        capital_required=capital,
        multiplier=multiplier,
    )

    print(calculate_stats(perc_return))
    print(calculate_stats(perc_return), MONTH)

    print(
        calculate_minimum_capital(
            multiplier=multiplier,
            risk_target=risk_target_tau,
            fx=1,
            instrument_risk_ann_perc=instrument_risk_ann_perc[-1],
            price=current_price[-1],
        )
    )

    print(
        calculate_turnover(
            position_contracts_held, average_position=position_contracts_held
        )
    )
