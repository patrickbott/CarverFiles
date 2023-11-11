"""
This is the provided example python code for Chapter twenty of the book:
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
from chapter10 import get_data_dict_with_carry, calculate_forecast_for_carry
from chapter11 import get_fdm
from chapter18 import get_asset_class_for_instrument


def calculate_position_dict_with_forecast_from_function_applied(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    list_of_rules: list,
) -> dict:

    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_carry = dict(
        [
            (
                instrument_code,
                calculate_position_with_forecast_applied_from_function(
                    instrument_code,
                    average_position_contracts_dict=average_position_contracts_dict,
                    adjusted_prices_dict=adjusted_prices_dict,
                    std_dev_dict=std_dev_dict,
                    carry_prices_dict=carry_prices_dict,
                    list_of_rules=list_of_rules,
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return position_dict_with_carry


def calculate_position_with_forecast_applied_from_function(
    instrument_code: str,
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    list_of_rules: list,
) -> pd.Series:

    forecast = calculate_combined_forecast_from_functions(
        instrument_code=instrument_code,
        adjusted_prices_dict=adjusted_prices_dict,
        std_dev_dict=std_dev_dict,
        carry_prices_dict=carry_prices_dict,
        list_of_rules=list_of_rules,
    )

    return forecast * average_position_contracts_dict[instrument_code] / 10


def calculate_combined_forecast_from_functions(
    instrument_code: str,
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    list_of_rules: list,
) -> pd.Series:

    all_forecasts_as_list = [
        calculate_forecast_from_function(
            instrument_code=instrument_code,
            adjusted_prices_dict=adjusted_prices_dict,
            std_dev_dict=std_dev_dict,
            carry_prices_dict=carry_prices_dict,
            rule=rule,
        )
        for rule in list_of_rules
    ]

    ### NOTE: This assumes we are equally weighted across spans
    ### eg all forecast weights the same, equally weighted
    all_forecasts_as_df = pd.concat(all_forecasts_as_list, axis=1)
    average_forecast = all_forecasts_as_df.mean(axis=1)

    ## apply an FDM
    rule_count = len(list_of_rules)
    fdm = get_fdm(rule_count)
    scaled_forecast = average_forecast * fdm
    capped_forecast = scaled_forecast.clip(-20, 20)

    return capped_forecast


def calculate_forecast_from_function(
    instrument_code: str,
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    rule: dict,
) -> pd.Series:

    rule_copy = copy(rule)
    rule_function = rule_copy.pop("function")
    scalar = rule_copy.pop("scalar")
    rule_args = rule_copy

    forecast_value = rule_function(
        instrument_code=instrument_code,
        adjusted_prices_dict=adjusted_prices_dict,
        std_dev_dict=std_dev_dict,
        carry_prices_dict=carry_prices_dict,
        **rule_args
    )

    return forecast_value * scalar


def relative_carry(
    instrument_code: str,
    adjusted_prices_dict: dict,  ## not used
    std_dev_dict: dict,
    carry_prices_dict: dict,
    span: int = 90,
) -> pd.Series:

    carry_forecast = calculate_forecast_for_carry(
        stdev_ann_perc=std_dev_dict[instrument_code],
        carry_price=carry_prices_dict[instrument_code],
        span=span,
    )

    median_forecast = median_carry_for_instrument_in_asset_class(
        instrument_code=instrument_code,
        std_dev_dict=std_dev_dict,
        carry_prices_dict=carry_prices_dict,
    )
    median_forecast_indexed = median_forecast.reindex(carry_forecast.index).ffill()

    relative_carry_forecast = carry_forecast - median_forecast_indexed
    relative_carry_forecast[relative_carry_forecast == 0] = np.nan

    return relative_carry_forecast


def median_carry_for_instrument_in_asset_class(
    instrument_code: str, std_dev_dict: dict, carry_prices_dict: dict, span: int = 90
) -> pd.Series:

    asset_class = get_asset_class_for_instrument(
        instrument_code, asset_class_groupings=asset_class_groupings
    )

    median_carry = median_carry_for_asset_class(
        asset_class,
        std_dev_dict=std_dev_dict,
        carry_prices_dict=carry_prices_dict,
        asset_class_groupings=asset_class_groupings,
        span=span,
    )

    return median_carry


def median_carry_for_asset_class(
    asset_class: str,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    asset_class_groupings: dict,
    span: int = 90,
) -> pd.Series:

    list_of_instruments = asset_class_groupings[asset_class]
    all_carry_forecasts = [
        calculate_forecast_for_carry(
            stdev_ann_perc=std_dev_dict[instrument_code],
            carry_price=carry_prices_dict[instrument_code],
            span=span,
        )
        for instrument_code in list_of_instruments
    ]
    all_carry_forecasts_pd = pd.concat(all_carry_forecasts, axis=1)
    median_carry = all_carry_forecasts_pd.median(axis=1)

    return median_carry


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

    asset_class_groupings = dict(bonds=["us2", "us10"], stocks=["sp500", "eurostx"])
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
    list_of_rules = [dict(function=relative_carry, span=90, scalar=50)]

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
