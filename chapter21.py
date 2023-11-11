"""
This is the provided example python code for Chapter twenty one of the book:
 "Advanced Futures Trading Strategies", by Robert Carver
 https://www.systematicmoney.org/advanced-futures

This code is copyright, Robert Carver 2022.
Shared under https://www.gnu.org/licenses/gpl-3.0.en.html
You may copy, modify, and share this code as long as this header is retained, and you disclose that it has been edited.
This code comes with no warranty, is not guaranteed to be accurate, and the author is not responsible for any losses that may result from it’s use.

Results may not match the book exactly as different data may be used
Results may be different from the corresponding spreadsheet as methods may be slightly different

"""
from copy import copy

import pandas as pd
import numpy as np

## Next two lines are optional depending on your IDE
import matplotlib

matplotlib.use("TkAgg")

from chapter1 import BUSINESS_DAYS_IN_YEAR
from chapter4 import (
    create_fx_series_given_adjusted_prices_dict,
    calculate_variable_standard_deviation_for_risk_targeting_from_dict,
    calculate_position_series_given_variable_risk_for_dict,
)

from chapter5 import calculate_perc_returns_for_dict_with_costs
from chapter8 import apply_buffering_to_position_dict
from chapter10 import get_data_dict_with_carry

from chapter20 import calculate_position_dict_with_forecast_from_function_applied


def breakout(
    instrument_code: str,
    adjusted_prices_dict: dict,
    std_dev_dict: dict,  ## not used
    carry_prices_dict: dict,  ## not used
    scalar: float = 1.0,
    horizon: int = 10,
) -> pd.Series:

    breakout_forecast = calculate_forecast_for_breakout(
        adjusted_price=adjusted_prices_dict[instrument_code],
        horizon=horizon,
        scalar=scalar,
    )

    return breakout_forecast


def calculate_forecast_for_breakout(
    adjusted_price: pd.Series, horizon: int = 10, scalar: float = 1.0
) -> pd.Series:

    max_price = adjusted_price.rolling(horizon, min_periods=1).max()
    min_price = adjusted_price.rolling(horizon, min_periods=1).min()
    mean_price = (max_price + min_price) / 2
    raw_forecast = 40 * (adjusted_price - mean_price) / (max_price - min_price)
    smoothed_forecast = raw_forecast.ewm(span=int(np.ceil(horizon / 4))).mean()

    return smoothed_forecast * scalar


if __name__ == "__main__":
    ## Get the files from:
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/sp500.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/sp500_carry.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/us10.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/us10_carry.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/eur_fx.csv

    (
        adjusted_prices_dict,
        current_prices_dict,
        carry_prices_dict,
    ) = get_data_dict_with_carry(["sp500", "us10"])

    multipliers = dict(sp500=5, us10=1000)
    risk_target_tau = 0.2
    fx_series_dict = create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict)

    capital = 2000000

    idm = 1.3
    instrument_weights = dict(sp500=0.5, us10=0.5)

    cost_per_contract_dict = dict(
        sp500=0.875,
        us10=9.5,
    )

    std_dev_dict = calculate_variable_standard_deviation_for_risk_targeting_from_dict(
        adjusted_prices=adjusted_prices_dict, current_prices=current_prices_dict
    )

    average_position_contracts_dict = (
        calculate_position_series_given_variable_risk_for_dict(
            capital=capital,
            risk_target_tau=risk_target_tau,
            idm=idm,
            weights=instrument_weights,
            std_dev_dict=std_dev_dict,
            fx_series_dict=fx_series_dict,
            multipliers=multipliers,
        )
    )

    ## Assumes equal forecast weights and we use all rules for both instruments
    ## Note this will not use the table 96 FDM values, but the more general ones

    list_of_rules = [
        dict(function=breakout, scalar=0.6, horizon=10),
        dict(function=breakout, scalar=0.67, horizon=20),
        dict(function=breakout, scalar=0.7, horizon=40),
        dict(function=breakout, scalar=0.73, horizon=80),
        dict(function=breakout, scalar=0.74, horizon=160),
        dict(function=breakout, scalar=0.74, horizon=320),
    ]

    position_contracts_dict = (
        calculate_position_dict_with_forecast_from_function_applied(
            adjusted_prices_dict=adjusted_prices_dict,
            carry_prices_dict=carry_prices_dict,
            std_dev_dict=std_dev_dict,
            average_position_contracts_dict=average_position_contracts_dict,
            list_of_rules=list_of_rules,
        )
    )

    buffered_position_dict = apply_buffering_to_position_dict(
        position_contracts_dict=position_contracts_dict,
        average_position_contracts_dict=average_position_contracts_dict,
    )

    perc_return_dict = calculate_perc_returns_for_dict_with_costs(
        position_contracts_dict=buffered_position_dict,
        fx_series=fx_series_dict,
        multipliers=multipliers,
        capital=capital,
        adjusted_prices=adjusted_prices_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )
