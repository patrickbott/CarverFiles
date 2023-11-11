"""
This is the provided example python code for Chapter eighteen of the book:
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

from chapter4 import (
    create_fx_series_given_adjusted_prices_dict,
    calculate_variable_standard_deviation_for_risk_targeting_from_dict,
    calculate_position_series_given_variable_risk_for_dict,
)


from chapter5 import calculate_perc_returns_for_dict_with_costs
from chapter8 import apply_buffering_to_position_dict
from chapter10 import get_data_dict_with_carry
from chapter11 import calculate_position_dict_with_forecast_applied
from chapter17 import calculate_normalised_price_dict


def calculate_asset_class_price_dict(
    normalised_price_dict: dict, asset_class_groupings: dict
):

    list_of_instruments = list(normalised_price_dict.keys())
    asset_class_price_dict = dict(
        [
            (
                instrument_code,
                calculate_asset_prices_for_instrument(
                    instrument_code,
                    normalised_price_dict=normalised_price_dict,
                    asset_class_groupings=asset_class_groupings,
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return asset_class_price_dict


def calculate_asset_prices_for_instrument(
    instrument_code: str, normalised_price_dict: dict, asset_class_groupings: dict
) -> pd.Series:

    asset_class = get_asset_class_for_instrument(
        instrument_code, asset_class_groupings=asset_class_groupings
    )

    return get_normalised_price_for_asset_class(
        asset_class=asset_class,
        asset_class_groupings=asset_class_groupings,
        normalised_price_dict=normalised_price_dict,
    )


def get_asset_class_for_instrument(
    instrument_code: str, asset_class_groupings: dict
) -> str:

    possible_asset_classes = list(asset_class_groupings.keys())
    asset_class = [
        asset
        for asset in possible_asset_classes
        if instrument_code in asset_class_groupings[asset]
    ][0]

    return asset_class


def get_normalised_price_for_asset_class(
    asset_class: str, normalised_price_dict: dict, asset_class_groupings: dict
) -> pd.Series:

    # Wasteful rerunning this for each instrument but makes code simpler
    instruments_in_asset_class = asset_class_groupings[asset_class]
    list_of_normalised_prices_over_asset_class = [
        normalised_price_dict[instrument_code]
        for instrument_code in instruments_in_asset_class
    ]
    normalised_prices_over_asset_class = pd.concat(
        list_of_normalised_prices_over_asset_class, axis=1
    ).ffill()
    normalised_returns_over_asset_class = normalised_prices_over_asset_class.diff()
    average_normalised_return_over_asset_class = (
        normalised_returns_over_asset_class.mean(axis=1)
    )

    asset_class_price = average_normalised_return_over_asset_class.cumsum()

    return asset_class_price


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

    normalised_price_dict = calculate_normalised_price_dict(
        adjusted_prices_dict=adjusted_prices_dict, std_dev_dict=std_dev_dict
    )

    asset_class_price_dict = calculate_asset_class_price_dict(
        normalised_price_dict=normalised_price_dict,
        asset_class_groupings=asset_class_groupings,
    )

    ## Assumes equal forecast weights and we use all rules for both instruments
    rules_spec = [
        dict(function="ewmac", fast_span=16),
        dict(function="ewmac", fast_span=32),
        dict(function="ewmac", fast_span=64),
    ]
    position_contracts_dict = calculate_position_dict_with_forecast_applied(
        adjusted_prices_dict=asset_class_price_dict,
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
