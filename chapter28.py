"""
This is the provided example python code for Chapter twenty eight of the book:
 "Advanced Futures Trading Strategies", by Robert Carver
 https://www.systematicmoney.org/advanced-futures

This code is copyright, Robert Carver 2022.
Shared under https://www.gnu.org/licenses/gpl-3.0.en.html
You may copy, modify, and share this code as long as this header is retained, and you disclose that it has been edited.
This code comes with no warranty, is not guaranteed to be accurate or to match the methodology in the book, and the author is not responsible for any losses that may result from it’s use.

Results may not match the book exactly as different data may be used and the code may contain errors
Results may be different from the corresponding spreadsheet as methods may be slightly different

"""
## Next two lines are optional depending on your IDE
import matplotlib

matplotlib.use("TkAgg")
import pandas as pd
import datetime
from chapter4 import (
    create_fx_series_given_adjusted_prices_dict,
    calculate_variable_standard_deviation_for_risk_targeting_from_dict,
    calculate_position_series_given_variable_risk_for_dict,
)


from chapter5 import calculate_perc_returns_for_dict_with_costs
from chapter8 import apply_buffering_to_position_dict
from chapter10 import get_data_dict_with_carry
from chapter11 import calculate_position_dict_with_forecast_applied


def transform_into_RV_prices(
    adjusted_prices_dict: dict,
    carry_prices_dict: dict,
    multipliers: dict,
    fx_series_dict: dict,
    current_prices_dict: dict,
    instrument_code_list: list,
    ratio_list: list,
    rv_instrument_name: str,
    start_date: datetime.datetime = None,
) -> dict:

    ## instrument_code_list for spreads we have b,a; for triplets we have b,a,c
    # ratio_list is eithier -1,R for spreads or -1, X, Y for triplets

    if start_date is None:
        start_date = datetime.datetime(1970, 1, 1)

    instrument_code_b = instrument_code_list[0]
    weighted_adj_prices_as_list = [
        adjusted_prices_dict[instrument_code][start_date:] * ratio_list[i]
        for i, instrument_code in enumerate(instrument_code_list)
    ]

    weighted_adj_prices_as_df = pd.concat(weighted_adj_prices_as_list, axis=1)
    new_adj_price = weighted_adj_prices_as_df.sum(axis=1, skipna=False)
    new_adj_price = new_adj_price.dropna()

    weighted_current_prices_as_list = [
        current_prices_dict[instrument_code] * ratio_list[i]
        for i, instrument_code in enumerate(instrument_code_list)
    ]

    weighted_current_prices_as_df = pd.concat(weighted_current_prices_as_list, axis=1)
    new_current_price = weighted_current_prices_as_df.sum(axis=1, skipna=False)
    new_current_price = new_current_price.dropna()
    new_current_price = new_current_price.reindex(new_adj_price.index, method="ffill")

    weighted_carry_prices_as_list = [
        carry_prices_dict[instrument_code]["CARRY"] * ratio_list[i]
        for i, instrument_code in enumerate(instrument_code_list)
    ]

    weighted_carry_prices_as_df = pd.concat(weighted_carry_prices_as_list, axis=1)
    carry_price = weighted_carry_prices_as_df.sum(axis=1, skipna=False)

    weighted_priced_contract_prices_as_list = [
        carry_prices_dict[instrument_code]["PRICE"] * ratio_list[i]
        for i, instrument_code in enumerate(instrument_code_list)
    ]

    weighted_priced_contract_prices_as_df = pd.concat(
        weighted_priced_contract_prices_as_list, axis=1
    )
    priced_contract_price = weighted_priced_contract_prices_as_df.sum(
        axis=1, skipna=False
    )

    new_carry_df = pd.DataFrame(dict(PRICE=priced_contract_price, CARRY=carry_price))

    new_carry_price_contract = carry_prices_dict[instrument_code_b][
        "PRICE_CONTRACT"
    ].reindex(new_carry_df.index)
    new_carry_carry_contract = carry_prices_dict[instrument_code_b][
        "CARRY_CONTRACT"
    ].reindex(new_carry_df.index)

    ## ensures we will only keep days where both prices available
    new_carry_df["PRICE_CONTRACT"] = new_carry_price_contract
    new_carry_df["CARRY_CONTRACT"] = new_carry_carry_contract

    new_carry_df = new_carry_df.dropna()
    new_carry_df = new_carry_df.reindex(new_adj_price.index, method="ffill")

    """
    adjusted_prices_dict,
    carry_prices_dict,
    multipliers,
    fx_series_dict
    """

    return dict(
        adjusted_prices_dict={rv_instrument_name: new_adj_price},
        carry_prices_dict={rv_instrument_name: new_carry_df},
        current_prices_dict={rv_instrument_name: new_current_price},
        multipliers={rv_instrument_name: multipliers[instrument_code_b]},
        fx_series_dict={rv_instrument_name: fx_series_dict[instrument_code_b]},
    )


if __name__ == "__main__":
    ## Get the files from:
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/us10.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/us10_carry.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/us5_carry.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/us5.csv
    (
        adjusted_prices_dict,
        current_prices_dict,
        carry_prices_dict,
    ) = get_data_dict_with_carry(instrument_list=["us5", "us10"])

    multipliers = dict(us5=1000, us10=1000)
    risk_target_tau = 0.1
    fx_series_dict = create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict)

    instrument_code_list = ["us10", "us5"]
    ratio_list = [-1, 1.5]
    rv_instrument_name = "1.5us10_us5"

    new_data = transform_into_RV_prices(
        adjusted_prices_dict=adjusted_prices_dict,
        carry_prices_dict=carry_prices_dict,
        multipliers=multipliers,
        fx_series_dict=fx_series_dict,
        current_prices_dict=current_prices_dict,
        instrument_code_list=instrument_code_list,
        ratio_list=ratio_list,
        rv_instrument_name=rv_instrument_name,
    )

    capital = 2000000

    idm = 1.0
    instrument_weights = {rv_instrument_name: 1.0}
    cost_per_contract_dict = {rv_instrument_name: 19.75}

    std_dev_dict = calculate_variable_standard_deviation_for_risk_targeting_from_dict(
        adjusted_prices=new_data["adjusted_prices_dict"],
        current_prices=new_data["current_prices_dict"],
        use_perc_returns=False,  ## otherwise will cause problems
    )

    average_position_contracts_dict = (
        calculate_position_series_given_variable_risk_for_dict(
            capital=capital,
            risk_target_tau=risk_target_tau,
            idm=idm,
            weights=instrument_weights,
            std_dev_dict=std_dev_dict,
            fx_series_dict=new_data["fx_series_dict"],
            multipliers=new_data["multipliers"],
        )
    )

    ## Assumes equal forecast weights and we use all rules for both instruments
    rules_spec = [
        dict(function="carry", span=5),
        dict(function="carry", span=20),
        dict(function="carry", span=60),
        dict(function="carry", span=120),
        dict(function="ewmac", fast_span=16),
        dict(function="ewmac", fast_span=32),
        dict(function="ewmac", fast_span=64),
    ]
    position_contracts_dict = calculate_position_dict_with_forecast_applied(
        adjusted_prices_dict=new_data["adjusted_prices_dict"],
        carry_prices_dict=new_data["carry_prices_dict"],
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
        fx_series=new_data["fx_series_dict"],
        multipliers=new_data["multipliers"],
        capital=capital,
        adjusted_prices=new_data["adjusted_prices_dict"],
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )
