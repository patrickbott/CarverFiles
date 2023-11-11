"""
This is the provided example python code for Chapter thirteen of the book:
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
from chapter3 import standardDeviation
from chapter4 import (
    create_fx_series_given_adjusted_prices_dict,
    calculate_variable_standard_deviation_for_risk_targeting_from_dict,
    calculate_position_series_given_variable_risk_for_dict,
)

from chapter5 import calculate_perc_returns_for_dict_with_costs
from chapter7 import calculate_risk_adjusted_forecast_for_ewmac
from chapter8 import apply_buffering_to_position_dict
from chapter10 import calculate_smoothed_carry, get_data_dict_with_carry
from chapter11 import get_fdm


def calculate_position_dict_with_forecast_and_vol_scalar_applied(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    rule_spec: list,
) -> dict:

    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_carry = dict(
        [
            (
                instrument_code,
                calculate_position_with_forecast_and_vol_scalar_applied(
                    average_position=average_position_contracts_dict[instrument_code],
                    stdev_ann_perc=std_dev_dict[instrument_code],
                    carry_price=carry_prices_dict[instrument_code],
                    adjusted_price=adjusted_prices_dict[instrument_code],
                    rule_spec=rule_spec,
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return position_dict_with_carry


def calculate_position_with_forecast_and_vol_scalar_applied(
    average_position: pd.Series,
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    adjusted_price: pd.Series,
    rule_spec: list,
) -> pd.Series:

    forecast = calculate_combined_forecast_with_vol_scalar_applied(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        carry_price=carry_price,
        rule_spec=rule_spec,
    )

    return forecast * average_position / 10


def calculate_combined_forecast_with_vol_scalar_applied(
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    adjusted_price: pd.Series,
    rule_spec: list,
) -> pd.Series:

    all_forecasts_as_list = [
        calculate_forecast_with_vol_scalar_applied(
            adjusted_price=adjusted_price,
            stdev_ann_perc=stdev_ann_perc,
            carry_price=carry_price,
            rule=rule,
        )
        for rule in rule_spec
    ]

    ### NOTE: This assumes we are equally weighted across spans
    ### eg all forecast weights the same, equally weighted
    all_forecasts_as_df = pd.concat(all_forecasts_as_list, axis=1)
    average_forecast = all_forecasts_as_df.mean(axis=1)

    ## apply an FDM
    rule_count = len(rule_spec)
    fdm = get_fdm(rule_count)
    scaled_forecast = average_forecast * fdm
    capped_forecast = scaled_forecast.clip(-20, 20)

    return capped_forecast


def calculate_forecast_with_vol_scalar_applied(
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    adjusted_price: pd.Series,
    rule: dict,
) -> pd.Series:

    if rule["function"] == "carry":
        span = rule["span"]
        forecast = calculate_forecast_for_carry_with_optional_vol_scaling(
            stdev_ann_perc=stdev_ann_perc, carry_price=carry_price, span=span
        )

    elif rule["function"] == "ewmac":
        fast_span = rule["fast_span"]
        forecast = calculate_forecast_for_ewmac_with_optional_vol_scaling(
            adjusted_price=adjusted_price,
            stdev_ann_perc=stdev_ann_perc,
            fast_span=fast_span,
        )

    else:
        raise Exception("Rule %s not recognised!" % rule["function"])

    return forecast


def calculate_forecast_for_carry_with_optional_vol_scaling(
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    span: int,
):

    smooth_carry = calculate_smoothed_carry(
        stdev_ann_perc=stdev_ann_perc,
        carry_price=carry_price,
        span=span,
    )
    if APPLY_VOL_REGIME_TO_CARRY:
        smooth_carry = apply_vol_regime_to_forecast(
            smooth_carry, stdev_ann_perc=stdev_ann_perc
        )
        scaled_carry = smooth_carry * 23
    else:
        scaled_carry = smooth_carry * 30

    capped_carry = scaled_carry.clip(-20, 20)

    return capped_carry


def calculate_forecast_for_ewmac_with_optional_vol_scaling(
    adjusted_price: pd.Series,
    stdev_ann_perc: standardDeviation,
    fast_span: int = 64,
):

    scalar_dict = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1}
    risk_adjusted_ewmac = calculate_risk_adjusted_forecast_for_ewmac(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_span=fast_span,
    )
    if APPLY_VOL_REGIME_TO_EWMAC:
        risk_adjusted_ewmac = apply_vol_regime_to_forecast(
            risk_adjusted_ewmac, stdev_ann_perc=stdev_ann_perc
        )

    forecast_scalar = scalar_dict[fast_span]
    scaled_ewmac = risk_adjusted_ewmac * forecast_scalar
    capped_ewmac = scaled_ewmac.clip(-20, 20)

    return capped_ewmac


def apply_vol_regime_to_forecast(
    scaled_forecast: pd.Series, stdev_ann_perc: pd.Series
) -> pd.Series:

    smoothed_vol_attenuation = get_attenuation(scaled_forecast)
    return scaled_forecast * smoothed_vol_attenuation


def get_attenuation(stdev_ann_perc: standardDeviation) -> pd.Series:
    normalised_vol = calculate_normalised_vol(stdev_ann_perc)
    normalised_vol_q = quantile_of_points_in_data_series(normalised_vol)
    vol_attenuation = normalised_vol_q.apply(multiplier_function)

    smoothed_vol_attenuation = vol_attenuation.ewm(span=10).mean()

    return smoothed_vol_attenuation


def multiplier_function(vol_quantile):
    if np.isnan(vol_quantile):
        return 1.0

    return 2 - 1.5 * vol_quantile


def calculate_normalised_vol(stdev_ann_perc: standardDeviation) -> pd.Series:
    ten_year_averages = stdev_ann_perc.rolling(2500, min_periods=10).mean()

    return stdev_ann_perc / ten_year_averages


def quantile_of_points_in_data_series(data_series):
    ## With thanks to https://github.com/PurpleHazeIan for this implementation
    numpy_series = np.array(data_series)
    results = []

    for irow in range(len(data_series)):
        current_value = numpy_series[irow]
        count_less_than = (numpy_series < current_value)[:irow].sum()
        results.append(count_less_than / (irow + 1))

    results_series = pd.Series(results, index=data_series.index)
    return results_series


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

    ## Assumes equal forecast weights and we use all rules for both instruments
    rules_spec = [
        dict(function="carry", span=60),
        dict(function="ewmac", fast_span=16),
        dict(function="ewmac", fast_span=32),
        dict(function="ewmac", fast_span=64),
    ]

    APPLY_VOL_REGIME_TO_EWMAC = True
    APPLY_VOL_REGIME_TO_CARRY = False

    position_contracts_dict = (
        calculate_position_dict_with_forecast_and_vol_scalar_applied(
            adjusted_prices_dict=adjusted_prices_dict,
            carry_prices_dict=carry_prices_dict,
            std_dev_dict=std_dev_dict,
            average_position_contracts_dict=average_position_contracts_dict,
            rule_spec=rules_spec,
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
