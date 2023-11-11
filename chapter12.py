"""
This is the provided example python code for Chapter twelve of the book:
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
from copy import copy

from chapter1 import calculate_stats, BUSINESS_DAYS_IN_YEAR
from chapter3 import standardDeviation
from chapter4 import (
    get_data_dict,
    create_fx_series_given_adjusted_prices_dict,
    calculate_variable_standard_deviation_for_risk_targeting_from_dict,
    calculate_position_series_given_variable_risk_for_dict,
)

from chapter5 import calculate_perc_returns_for_dict_with_costs
from chapter8 import apply_buffering_to_position_dict
from chapter5 import ewmac


def calculate_position_dict_with_multiple_trend_forecast_applied_and_adjustment(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    fast_spans: list,
) -> dict:

    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_trend_filter = dict(
        [
            (
                instrument_code,
                calculate_position_with_multiple_trend_forecast_applied_and_adjustment(
                    adjusted_prices_dict[instrument_code],
                    average_position_contracts_dict[instrument_code],
                    stdev_ann_perc=std_dev_dict[instrument_code],
                    fast_spans=fast_spans,
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return position_dict_with_trend_filter


def calculate_position_with_multiple_trend_forecast_applied_and_adjustment(
    adjusted_price: pd.Series,
    average_position: pd.Series,
    stdev_ann_perc: standardDeviation,
    fast_spans: list,
) -> pd.Series:

    forecast = calculate_combined_ewmac_forecast_and_adjustment(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_spans=fast_spans,
    )

    return forecast * average_position / 10


def calculate_combined_ewmac_forecast_and_adjustment(
    adjusted_price: pd.Series, stdev_ann_perc: standardDeviation, fast_spans: list
) -> pd.Series:

    all_forecasts_as_list = [
        calculate_forecast_for_ewmac_and_adjustment(
            adjusted_price=adjusted_price,
            stdev_ann_perc=stdev_ann_perc,
            fast_span=fast_span,
        )
        for fast_span in fast_spans
    ]

    ### NOTE: This assumes we are equally weighted across spans
    ### eg all forecast weights the same, equally weighted
    all_forecasts_as_df = pd.concat(all_forecasts_as_list, axis=1)
    average_forecast = all_forecasts_as_df.mean(axis=1)

    ## apply an FDM
    rule_count = len(fast_spans)
    FDM_DICT = {1: 1.0, 2: 1.03, 3: 1.08, 4: 1.13, 5: 1.19, 6: 1.26}
    fdm = FDM_DICT[rule_count]

    scaled_forecast = average_forecast * fdm
    capped_forecast = scaled_forecast.clip(-20, 20)

    return capped_forecast


def calculate_forecast_for_ewmac_and_adjustment(
    adjusted_price: pd.Series, stdev_ann_perc: standardDeviation, fast_span: int = 64
):

    scalar_dict = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1}
    ewmac_values = ewmac(adjusted_price, fast_span=fast_span, slow_span=fast_span * 4)
    daily_price_vol = stdev_ann_perc.daily_risk_price_terms()
    risk_adjusted_ewmac = ewmac_values / daily_price_vol
    forecast_scalar = scalar_dict[fast_span]
    scaled_ewmac = risk_adjusted_ewmac * forecast_scalar

    if fast_span == 2:
        capped_ewmac = double_v(scaled_ewmac)
    elif fast_span == 4 or fast_span == 64:
        capped_ewmac = scale_and_cap(scaled_ewmac)
    else:
        capped_ewmac = scaled_ewmac.clip(-20, 20)

    return capped_ewmac


def double_v(scaled_forecast: pd.Series) -> pd.Series:
    new_forecast = copy(scaled_forecast)
    new_forecast[scaled_forecast > 20] = 0
    new_forecast[scaled_forecast < -20] = 0
    new_forecast[(scaled_forecast >= -20) & (scaled_forecast < -10)] = (
        new_forecast[(scaled_forecast >= -20) & (scaled_forecast < -10)] * -2
    ) - 40
    new_forecast[(scaled_forecast >= -10) & (scaled_forecast < 10)] = (
        new_forecast[(scaled_forecast >= -10) & (scaled_forecast < +10)] * 2
    )

    new_forecast[(scaled_forecast >= 10) & (scaled_forecast < 20)] = (
        new_forecast[(scaled_forecast >= 10) & (scaled_forecast < +20)] * -2
    ) + 40

    return new_forecast


def scale_and_cap(scaled_forecast: pd.Series) -> pd.Series:
    rescaled_forecast = 1.25 * scaled_forecast
    capped_forecast = rescaled_forecast.clip(-1.25 * 15, 1.25 * 15)

    return capped_forecast


if __name__ == "__main__":
    ## Get the files from:
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/sp500.csv
    # and https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/US10.csv
    adjusted_prices_dict, current_prices_dict = get_data_dict()

    multipliers = dict(sp500=5, us10=1000)
    risk_target_tau = 0.2
    fx_series_dict = create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict)

    capital = 1000000

    idm = 1.5
    instrument_weights = dict(sp500=0.5, us10=0.5)
    cost_per_contract_dict = dict(sp500=0.875, us10=5)

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
            current_prices=current_prices_dict,
            fx_series_dict=fx_series_dict,
            multipliers=multipliers,
        )
    )

    ## We use all spans here for both instruments
    ## In reality we would need to check costs and turnover
    fast_spans = [2, 4, 8, 16, 32, 64]
    position_contracts_dict = (
        calculate_position_dict_with_multiple_trend_forecast_applied_and_adjustment(
            adjusted_prices_dict=adjusted_prices_dict,
            average_position_contracts_dict=average_position_contracts_dict,
            current_prices_dict=current_prices_dict,
            std_dev_dict=std_dev_dict,
            fast_spans=fast_spans,
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
        current_prices_dict=current_prices_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )

    print(calculate_stats(perc_return_dict["us10"]))
