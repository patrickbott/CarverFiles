"""
This is the provided example python code for Chapter four of the book:
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

import pandas as pd
import numpy as np

from chapter1 import pd_readcsv, calculate_stats, MONTH
from chapter3 import (
    standardDeviation,
    calculate_position_series_given_variable_risk,
    calculate_perc_returns,
)

INSTRUMENT_LIST = ["sp500", "us10"]


def get_data_dict():

    all_data = dict(
        [
            (instrument_code, pd_readcsv("%s.csv" % instrument_code))
            for instrument_code in INSTRUMENT_LIST
        ]
    )

    adjusted_prices = dict(
        [
            (instrument_code, data_for_instrument.adjusted)
            for instrument_code, data_for_instrument in all_data.items()
        ]
    )

    current_prices = dict(
        [
            (instrument_code, data_for_instrument.underlying)
            for instrument_code, data_for_instrument in all_data.items()
        ]
    )

    return adjusted_prices, current_prices


fx_dict = dict(eurostx="eur")


def create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict: dict) -> dict:
    fx_series_dict = dict(
        [
            (
                instrument_code,
                create_fx_series_given_adjusted_prices(
                    instrument_code, adjusted_prices
                ),
            )
            for instrument_code, adjusted_prices in adjusted_prices_dict.items()
        ]
    )
    return fx_series_dict


def create_fx_series_given_adjusted_prices(
    instrument_code: str, adjusted_prices: pd.Series
) -> pd.Series:

    currency_for_instrument = fx_dict.get(instrument_code, "usd")
    if currency_for_instrument == "usd":
        return pd.Series(1, index=adjusted_prices.index)  ## FX rate, 1 for USD / USD

    fx_prices = get_fx_prices(currency_for_instrument)
    fx_prices_aligned = fx_prices.reindex(adjusted_prices.index).ffill()

    return fx_prices_aligned


def get_fx_prices(currency: str) -> pd.Series:
    prices_as_df = pd_readcsv("%s_fx.csv" % currency)
    return prices_as_df.squeeze()


def calculate_variable_standard_deviation_for_risk_targeting_from_dict(
    adjusted_prices: dict,
    current_prices: dict,
    use_perc_returns: bool = True,
    annualise_stdev: bool = True,
) -> dict:

    std_dev_dict = dict(
        [
            (
                instrument_code,
                standardDeviation(
                    adjusted_price=adjusted_prices[instrument_code],
                    current_price=current_prices[instrument_code],
                    use_perc_returns=use_perc_returns,
                    annualise_stdev=annualise_stdev,
                ),
            )
            for instrument_code in adjusted_prices.keys()
        ]
    )

    return std_dev_dict


def calculate_position_series_given_variable_risk_for_dict(
    capital: float,
    risk_target_tau: float,
    idm: float,
    weights: dict,
    fx_series_dict: dict,
    multipliers: dict,
    std_dev_dict: dict,
) -> dict:

    position_series_dict = dict(
        [
            (
                instrument_code,
                calculate_position_series_given_variable_risk(
                    capital=capital * idm * weights[instrument_code],
                    risk_target_tau=risk_target_tau,
                    multiplier=multipliers[instrument_code],
                    fx=fx_series_dict[instrument_code],
                    instrument_risk=std_dev_dict[instrument_code],
                ),
            )
            for instrument_code in std_dev_dict.keys()
        ]
    )

    return position_series_dict


def calculate_perc_returns_for_dict(
    position_contracts_dict: dict,
    adjusted_prices: dict,
    multipliers: dict,
    fx_series: dict,
    capital: float,
) -> dict:

    perc_returns_dict = dict(
        [
            (
                instrument_code,
                calculate_perc_returns(
                    position_contracts_held=position_contracts_dict[instrument_code],
                    adjusted_price=adjusted_prices[instrument_code],
                    multiplier=multipliers[instrument_code],
                    fx_series=fx_series[instrument_code],
                    capital_required=capital,
                ),
            )
            for instrument_code in position_contracts_dict.keys()
        ]
    )

    return perc_returns_dict


def aggregate_returns(perc_returns_dict: dict) -> pd.Series:
    both_returns = perc_returns_to_df(perc_returns_dict)
    agg = both_returns.sum(axis=1)
    return agg


def perc_returns_to_df(perc_returns_dict: dict) -> pd.DataFrame:
    both_returns = pd.concat(perc_returns_dict, axis=1)
    both_returns = both_returns.dropna(how="all")

    return both_returns


def minimum_capital_for_sub_strategy(
    multiplier: float,
    price: float,
    fx: float,
    instrument_risk_ann_perc: float,
    risk_target: float,
    idm: float,
    weight: float,
    contracts: int = 4,
):
    # (4 × Multiplier i × Price i, t × FX rate i, t × σ % i, t) ÷ (IDM × Weight i × τ)
    return (
        contracts
        * multiplier
        * price
        * fx
        * instrument_risk_ann_perc
        / (risk_target * idm * weight)
    )


if __name__ == "__main__":
    ## Get the files from:
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/sp500.csv
    # and https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/US10.csv
    adjusted_prices, current_prices = get_data_dict()

    multipliers = dict(sp500=5, us10=1000)
    risk_target_tau = 0.2

    fx_series_dict = create_fx_series_given_adjusted_prices_dict(adjusted_prices)

    capital = 1000000
    idm = 1.5
    instrument_weights = dict(sp500=0.5, us10=0.5)

    std_dev_dict = calculate_variable_standard_deviation_for_risk_targeting_from_dict(
        adjusted_prices=adjusted_prices,
        current_prices=current_prices,
        annualise_stdev=True,  ## can also be False if want to use daily price diff
        use_perc_returns=True,  ## can also be False if want to use daily price diff
    )

    position_contracts_dict = calculate_position_series_given_variable_risk_for_dict(
        capital=capital,
        risk_target_tau=risk_target_tau,
        idm=idm,
        weights=instrument_weights,
        std_dev_dict=std_dev_dict,
        fx_series_dict=fx_series_dict,
        multipliers=multipliers,
    )

    perc_return_dict = calculate_perc_returns_for_dict(
        position_contracts_dict=position_contracts_dict,
        fx_series=fx_series_dict,
        multipliers=multipliers,
        capital=capital,
        adjusted_prices=adjusted_prices,
    )

    print(calculate_stats(perc_return_dict["sp500"]))

    perc_return_agg = aggregate_returns(perc_return_dict)
    print(calculate_stats(perc_return_agg))

    instrument_code = "us10"
    print(
        minimum_capital_for_sub_strategy(
            multiplier=multipliers[instrument_code],
            risk_target=risk_target_tau,
            fx=fx_series_dict[instrument_code][-1],
            idm=idm,
            weight=instrument_weights[instrument_code],
            instrument_risk_ann_perc=std_dev_dict[instrument_code][-1],
            price=current_prices[instrument_code][-1],
        )
    )
