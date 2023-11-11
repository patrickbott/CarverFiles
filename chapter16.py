"""
This is the provided example python code for Chapter sixteen of the book:
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

from chapter1 import BUSINESS_DAYS_IN_YEAR
from chapter4 import (
    create_fx_series_given_adjusted_prices_dict,
    calculate_variable_standard_deviation_for_risk_targeting_from_dict,
    calculate_position_series_given_variable_risk_for_dict,
    aggregate_returns,
)

from chapter5 import calculate_perc_returns_for_dict_with_costs
from chapter8 import apply_buffering_to_position_dict
from chapter10 import get_data_dict_with_carry
from chapter11 import calculate_position_dict_with_forecast_applied


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
    rules_spec_ewmac = [
        dict(function="ewmac", fast_span=16),
        dict(function="ewmac", fast_span=32),
        dict(function="ewmac", fast_span=64),
    ]
    position_contracts_dict_ewmac = calculate_position_dict_with_forecast_applied(
        adjusted_prices_dict=adjusted_prices_dict,
        carry_prices_dict=carry_prices_dict,
        std_dev_dict=std_dev_dict,
        average_position_contracts_dict=average_position_contracts_dict,
        rule_spec=rules_spec_ewmac,
    )

    buffered_position_dict_ewmac = apply_buffering_to_position_dict(
        position_contracts_dict=position_contracts_dict_ewmac,
        average_position_contracts_dict=average_position_contracts_dict,
    )

    perc_return_dict_ewmac = calculate_perc_returns_for_dict_with_costs(
        position_contracts_dict=buffered_position_dict_ewmac,
        fx_series=fx_series_dict,
        multipliers=multipliers,
        capital=capital,
        adjusted_prices=adjusted_prices_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )

    perc_return_aggregated_ewmac = aggregate_returns(perc_return_dict_ewmac)

    rules_spec_carry = [
        dict(function="carry", span=5),
        dict(function="carry", span=20),
        dict(function="carry", span=60),
        dict(function="carry", span=120),
    ]
    position_contracts_dict_carry = calculate_position_dict_with_forecast_applied(
        adjusted_prices_dict=adjusted_prices_dict,
        carry_prices_dict=carry_prices_dict,
        std_dev_dict=std_dev_dict,
        average_position_contracts_dict=average_position_contracts_dict,
        rule_spec=rules_spec_carry,
    )

    buffered_position_dict_carry = apply_buffering_to_position_dict(
        position_contracts_dict=position_contracts_dict_carry,
        average_position_contracts_dict=average_position_contracts_dict,
    )

    perc_return_dict_carry = calculate_perc_returns_for_dict_with_costs(
        position_contracts_dict=buffered_position_dict_carry,
        fx_series=fx_series_dict,
        multipliers=multipliers,
        capital=capital,
        adjusted_prices=adjusted_prices_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )

    perc_return_aggregated_carry = aggregate_returns(perc_return_dict_carry)

    starting_portfolio = (
        perc_return_aggregated_ewmac * 0.6 + perc_return_aggregated_carry * 0.4
    )

    relative_performance = perc_return_aggregated_ewmac - perc_return_aggregated_carry
    rolling_12_month = relative_performance.rolling(BUSINESS_DAYS_IN_YEAR).mean()

    relative_performance = perc_return_aggregated_ewmac - perc_return_aggregated_carry
    rolling_12_month = (
        relative_performance.rolling(BUSINESS_DAYS_IN_YEAR).sum() / risk_target_tau
    )

    # W t = EWMA span=30 (min(1, max(0, 0.5 + RP t ÷ 2)))
    raw_weighting = 0.5 + rolling_12_month / 2
    clipped_weighting = raw_weighting.clip(lower=0, upper=1)
    smoothed_weighting = clipped_weighting.ewm(30).mean()

    weighted_portfolio = (
        perc_return_aggregated_ewmac * 0.6 * smoothed_weighting
        + perc_return_aggregated_carry * 0.4 * (1 - smoothed_weighting)
    )
