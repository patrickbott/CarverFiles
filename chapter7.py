"""
This is the provided example python code for Chapter seven of the book:
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

from scipy.stats import linregress
import pandas as pd

from chapter1 import calculate_stats, MONTH, BUSINESS_DAYS_IN_YEAR
from chapter3 import standardDeviation
from chapter4 import (
    get_data_dict,
    calculate_variable_standard_deviation_for_risk_targeting_from_dict,
    calculate_position_series_given_variable_risk_for_dict,
    create_fx_series_given_adjusted_prices_dict,
    aggregate_returns,
)
from chapter5 import ewmac, calculate_perc_returns_for_dict_with_costs
from chapter6 import long_only_returns


def calculate_position_dict_with_trend_forecast_applied(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    fast_span: int = 64,
) -> dict:

    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_trend_filter = dict(
        [
            (
                instrument_code,
                calculate_position_with_trend_forecast_applied(
                    adjusted_prices_dict[instrument_code],
                    average_position_contracts_dict[instrument_code],
                    stdev_ann_perc=std_dev_dict[instrument_code],
                    fast_span=fast_span,
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return position_dict_with_trend_filter


def calculate_position_with_trend_forecast_applied(
    adjusted_price: pd.Series,
    average_position: pd.Series,
    stdev_ann_perc: standardDeviation,
    fast_span: int = 64,
) -> pd.Series:

    forecast = calculate_forecast_for_ewmac(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_span=fast_span,
    )

    return forecast * average_position / 10


def calculate_forecast_for_ewmac(
    adjusted_price: pd.Series, stdev_ann_perc: standardDeviation, fast_span: int = 64
):

    scaled_ewmac = calculate_scaled_forecast_for_ewmac(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_span=fast_span,
    )
    capped_ewmac = scaled_ewmac.clip(-20, 20)

    return capped_ewmac


def calculate_scaled_forecast_for_ewmac(
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
    forecast_scalar = scalar_dict[fast_span]
    scaled_ewmac = risk_adjusted_ewmac * forecast_scalar

    return scaled_ewmac


def calculate_risk_adjusted_forecast_for_ewmac(
    adjusted_price: pd.Series,
    stdev_ann_perc: standardDeviation,
    fast_span: int = 64,
):

    ewmac_values = ewmac(adjusted_price, fast_span=fast_span, slow_span=fast_span * 4)
    daily_price_vol = stdev_ann_perc.daily_risk_price_terms()

    risk_adjusted_ewmac = ewmac_values / daily_price_vol

    return risk_adjusted_ewmac


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
        adjusted_prices=adjusted_prices_dict,
        current_prices=current_prices_dict,
        annualise_stdev=True,
        use_perc_returns=True,
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

    position_contracts_dict = calculate_position_dict_with_trend_forecast_applied(
        adjusted_prices_dict=adjusted_prices_dict,
        average_position_contracts_dict=average_position_contracts_dict,
        std_dev_dict=std_dev_dict,
        fast_span=64,
    )

    ## note doesn't include roll costs
    perc_return_dict = calculate_perc_returns_for_dict_with_costs(
        position_contracts_dict=position_contracts_dict,
        fx_series=fx_series_dict,
        multipliers=multipliers,
        capital=capital,
        adjusted_prices=adjusted_prices_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )

    print(calculate_stats(perc_return_dict["sp500"]))

    perc_return_agg = aggregate_returns(perc_return_dict)
    print(calculate_stats(perc_return_agg))

    long_only = long_only_returns(
        adjusted_prices_dict=adjusted_prices_dict,
        average_position_contracts_dict=average_position_contracts_dict,
        capital=capital,
        cost_per_contract_dict=cost_per_contract_dict,
        fx_series_dict=fx_series_dict,
        multipliers=multipliers,
        std_dev_dict=std_dev_dict,
    )

    results = linregress(long_only, perc_return_agg)
    print("Beta %f" % results.slope)
    daily_alpha = results.intercept
    print("Annual alpha %.2f%%" % (100 * daily_alpha * BUSINESS_DAYS_IN_YEAR))
