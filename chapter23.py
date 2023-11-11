"""
This is the provided example python code for Chapter twenty three of the book:
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

## Next two lines are optional depending on your IDE
import matplotlib

matplotlib.use("TkAgg")


from chapter1 import BUSINESS_DAYS_IN_YEAR
from chapter3 import standardDeviation
from chapter4 import (
    create_fx_series_given_adjusted_prices_dict,
    calculate_variable_standard_deviation_for_risk_targeting_from_dict,
    calculate_position_series_given_variable_risk_for_dict,
)

from chapter5 import calculate_perc_returns_for_dict_with_costs
from chapter7 import calculate_scaled_forecast_for_ewmac
from chapter8 import apply_buffering_to_position_dict
from chapter10 import get_data_dict_with_carry

from chapter20 import calculate_position_dict_with_forecast_from_function_applied


def accel(
    instrument_code: str,
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,  ## not used
    fast_span: int = 32,
    scalar: float = 7.27,
) -> pd.Series:

    accel_forecast = calculate_forecast_for_accel(
        adjusted_price=adjusted_prices_dict[instrument_code],
        stdev_ann_perc=std_dev_dict[instrument_code],
        scalar=scalar,
        fast_span=fast_span,
    )

    return accel_forecast


def calculate_forecast_for_accel(
    adjusted_price: pd.Series,
    stdev_ann_perc: standardDeviation,
    scalar: float,
    fast_span: int = 64,
) -> pd.Series:

    ## this returns a scaled forecast, but not capped
    ewmac_forecast = calculate_scaled_forecast_for_ewmac(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_span=fast_span,
    )

    accel_raw_forecast = ewmac_forecast - ewmac_forecast.shift(fast_span)
    scaled_accel_forecast = accel_raw_forecast * scalar
    capped_accel_forecast = scaled_accel_forecast.clip(-20, 20)

    return capped_accel_forecast


if __name__ == "__main__":
    ## Get the files from:
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/sp500.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/sp500_carry.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/eurostx.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/eurostx_carry.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/us10.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/us10_carry.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/us2.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/us2_carry.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/eur_fx.csv

    (
        adjusted_prices_dict,
        current_prices_dict,
        carry_prices_dict,
    ) = get_data_dict_with_carry(["sp500", "eurostx", "us10", "us2"])

    multipliers = dict(sp500=5, eurostx=10, us10=1000, us2=2000)
    risk_target_tau = 0.2
    fx_series_dict = create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict)

    capital = 2000000

    idm = 2
    instrument_weights = dict(sp500=0.25, eurostx=0.25, us10=0.25, us2=0.25)

    cost_per_contract_dict = dict(sp500=0.875, eurostx=6.8, us10=9.5, us2=5.5)

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
    list_of_rules = [
        dict(function=accel, fast_span=16, scalar=1.87),
        dict(function=accel, fast_span=16, scalar=1.90),
        dict(function=accel, fast_span=32, scalar=1.98),
        dict(function=accel, fast_span=64, scalar=2.05),
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
