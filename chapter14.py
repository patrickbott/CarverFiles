"""
This is the provided example python code for Chapter fourteen of the book:
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

## Next two lines are optional depending on your IDE
import matplotlib

matplotlib.use("TkAgg")

from chapter3 import standardDeviation
from chapter4 import (
    create_fx_series_given_adjusted_prices_dict,
    calculate_variable_standard_deviation_for_risk_targeting_from_dict,
    calculate_position_series_given_variable_risk_for_dict,
)

from chapter5 import calculate_perc_returns_for_dict_with_costs
from chapter8 import apply_buffering_to_position_dict
from chapter10 import get_data_dict_with_carry, calculate_annualised_carry
from chapter11 import calculate_position_dict_with_forecast_applied


def calculate_synthetic_spot_dict(
    adjusted_prices_dict: dict, carry_prices_dict: dict
) -> dict:

    list_of_instruments = list(adjusted_prices_dict.keys())
    synthetic_spot_dict = dict(
        [
            (
                instrument_code,
                calculate_synthetic_spot(
                    adjusted_prices_dict[instrument_code],
                    carry_price=carry_prices_dict[instrument_code],
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return synthetic_spot_dict


def calculate_synthetic_spot(
    adjusted_price: pd.Series, carry_price: pd.Series
) -> pd.Series:

    ann_carry = calculate_annualised_carry(carry_price)
    diff_index_in_years_as_pd = pd_series_of_diff_index_in_years(ann_carry)

    carry_per_period = diff_index_in_years_as_pd * ann_carry
    cum_carry = carry_per_period.cumsum()
    syn_spot = adjusted_price - cum_carry

    return syn_spot


def pd_series_of_diff_index_in_years(x: pd.Series):
    diff_index_in_years = get_annual_intervals_from_series(x)

    return pd.Series([0] + diff_index_in_years, x.index)


def get_annual_intervals_from_series(x: pd.Series):
    diff_index = x[1:].index - x[:-1].index
    diff_index_as_list = list(diff_index)
    diff_index_in_seconds = [
        index_item.total_seconds() for index_item in diff_index_as_list
    ]
    diff_index_in_years = [
        index_item_in_seconds / SECONDS_IN_YEAR
        for index_item_in_seconds in diff_index_in_seconds
    ]

    return diff_index_in_years


SECONDS_IN_YEAR = 60 * 60 * 24 * 365.25

if __name__ == "__main__":
    ## Get the files from:
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/sp500.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/sp500_carry.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/gas_carry.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/gas.csv
    (
        adjusted_prices_dict,
        current_prices_dict,
        carry_prices_dict,
    ) = get_data_dict_with_carry()

    multipliers = dict(sp500=5, gas=10000)
    risk_target_tau = 0.2
    fx_series_dict = create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict)

    capital = 2000000

    idm = 1.5
    instrument_weights = dict(sp500=0.5, gas=0.5)
    cost_per_contract_dict = dict(sp500=0.875, gas=15.3)

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

    synthetic_spot_dict = calculate_synthetic_spot_dict(
        adjusted_prices_dict, carry_prices_dict=carry_prices_dict
    )

    ## Assumes equal forecast weights and we use all rules for both instruments
    rules_spec = [
        dict(function="ewmac", fast_span=16),
        dict(function="ewmac", fast_span=32),
        dict(function="ewmac", fast_span=64),
    ]
    position_contracts_dict = calculate_position_dict_with_forecast_applied(
        adjusted_prices_dict=synthetic_spot_dict,
        carry_prices_dict=carry_prices_dict,
        std_dev_dict=std_dev_dict,
        average_position_contracts_dict=average_position_contracts_dict,
        rule_spec=rules_spec,
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
