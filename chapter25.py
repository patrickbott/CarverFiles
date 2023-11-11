"""
This is the provided example python code for Chapter twenty five of the book:
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

from chapter4 import (
    create_fx_series_given_adjusted_prices_dict,
    calculate_variable_standard_deviation_for_risk_targeting_from_dict,
    calculate_position_series_given_variable_risk_for_dict,
)

from chapter5 import calculate_perc_returns_for_dict_with_costs
from chapter10 import get_data_dict_with_carry
from chapter11 import calculate_position_dict_with_forecast_applied

## GET THE FOLLOWING FILE FROM:  https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/chapter25objects_and_functions.py
from chapter25objects_and_functions import *


def dynamically_optimise_positions(
    capital: float,
    fx_series_dict: dict,
    unrounded_position_contracts_dict: dict,
    multipliers: dict,
    std_dev_dict: dict,
    current_prices_dict: dict,
    adjusted_prices_dict: dict,
    cost_per_contract_dict: dict,
    algo_to_use,
) -> dict:

    data_for_optimisation = get_data_for_dynamic_optimisation(
        capital=capital,
        current_prices_dict=current_prices_dict,
        std_dev_dict=std_dev_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        fx_series_dict=fx_series_dict,
        adjusted_prices_dict=adjusted_prices_dict,
        multipliers=multipliers,
        unrounded_position_contracts_dict=unrounded_position_contracts_dict,
    )

    position_list = []
    common_index = data_for_optimisation.common_index
    previous_position = get_initial_positions(unrounded_position_contracts_dict)

    for relevant_date in common_index:
        data_for_single_period = get_data_for_relevant_date(
            relevant_date, data_for_optimisation=data_for_optimisation
        )

        optimal_positions = optimisation_for_single_period(
            previous_position=previous_position,
            data_for_single_period=data_for_single_period,
            algo_to_use=algo_to_use,
        )

        position_list.append(optimal_positions)
        previous_position = copy(optimal_positions)

    position_df = pd.DataFrame(position_list, index=common_index)

    ## single dataframe but we operate with a dict of series elsewhere
    positions_as_dict = from_df_to_dict_of_series(position_df)

    return positions_as_dict


def optimisation_for_single_period(
    previous_position: positionContracts,
    data_for_single_period: dataForSinglePeriod,
    algo_to_use,
) -> positionContracts:

    assets_with_data = which_assets_have_data(data_for_single_period)
    if len(assets_with_data) == 0:
        return previous_position

    assets_without_data = which_assets_without_data(
        data_for_single_period, assets_with_data=assets_with_data
    )

    data_for_single_period = data_for_single_period_with_valid_assets_only(
        data_for_single_period, assets_with_data=assets_with_data
    )

    previous_position = previous_position.with_selected_assets_only(assets_with_data)

    optimised_position = optimisation_for_single_period_with_valid_assets_only(
        previous_position=previous_position,
        data_for_single_period=data_for_single_period,
        algo_to_use=algo_to_use,
    )

    optimised_position_with_all_assets = (
        optimised_position.with_fill_for_missing_assets(assets_without_data)
    )

    return optimised_position_with_all_assets


def optimisation_for_single_period_with_valid_assets_only(
    previous_position: positionContracts,
    data_for_single_period: dataForSinglePeriod,
    algo_to_use,
) -> positionContracts:

    data_for_single_period_with_weights = (
        dataForSinglePeriodWithWeights.from_data_for_single_period(
            previous_position=previous_position,
            data_for_single_period=data_for_single_period,
        )
    )

    optimised_weights = optimisation_of_weight_for_single_period_with_valid_assets_only(
        data_for_single_period_with_weights, algo_to_use=algo_to_use
    )

    weights_per_contract = data_for_single_period_with_weights.weight_per_contract
    optimised_contracts = position_contracts_from_position_weights(
        optimised_weights, weights_per_contract=weights_per_contract
    )

    return optimised_contracts


def optimisation_of_weight_for_single_period_with_valid_assets_only(
    data_for_single_period_with_weights: dataForSinglePeriodWithWeights, algo_to_use
) -> positionWeights:

    data_for_single_period_as_np = (
        dataForSinglePeriodWithWeightsAsNp.from_data_for_single_period_with_weights(
            data_for_single_period_with_weights
        )
    )

    solution_as_np = algo_to_use(data_for_single_period_as_np)

    list_of_assets = list(
        data_for_single_period_with_weights.unrounded_optimal_position_weights.keys()
    )
    solution_as_weights = positionWeights.from_weights_and_keys(
        list_of_keys=list_of_assets, list_of_weights=list(solution_as_np)
    )

    return solution_as_weights


def greedy_algo_across_integer_values(
    data_for_single_period_as_np: dataForSinglePeriodWithWeightsAsNp,
) -> np.array:
    ## step 1
    weight_start = data_for_single_period_as_np.starting_weights

    current_best_value = evaluate_tracking_error(
        weight_start, data_for_single_period_as_np
    )
    current_best_solution = weight_start

    done = False

    while not done:
        ## step 3 loop
        (
            new_best_proposed_value,
            new_best_proposed_solution,
        ) = find_best_proposed_solution(
            current_best_solution=current_best_solution,
            current_best_value=current_best_value,
            data_for_single_period_as_np=data_for_single_period_as_np,
        )
        if new_best_proposed_value < current_best_value:
            # reached a new optimium
            # step 6
            current_best_value = new_best_proposed_value
            current_best_solution = new_best_proposed_solution
        else:
            # we can't do any better
            # step 7
            break

    return current_best_solution


def evaluate_tracking_error(
    weights: np.array, data_for_single_period_as_np: dataForSinglePeriodWithWeightsAsNp
):

    optimal_weights = data_for_single_period_as_np.unrounded_optimal_position_weights
    covariance = data_for_single_period_as_np.covariance_matrix

    return evaluate_tracking_error_for_weights(
        weights, optimal_weights, covariance=covariance
    )


def evaluate_tracking_error_for_weights(
    weights: np.array, other_weights, covariance: np.array
) -> float:

    solution_gap = weights - other_weights
    track_error_var = solution_gap.dot(covariance).dot(solution_gap)

    if track_error_var < 0:
        raise Exception("Negative covariance when optimising!")

    track_error_std = track_error_var**0.5

    return track_error_std


def find_best_proposed_solution(
    current_best_solution: np.array,
    current_best_value: float,
    data_for_single_period_as_np: dataForSinglePeriodWithWeightsAsNp,
) -> tuple:
    best_proposed_value = copy(current_best_value)
    best_proposed_solution = copy(current_best_solution)

    per_contract_value = data_for_single_period_as_np.weight_per_contract
    direction = data_for_single_period_as_np.direction_as_np

    count_assets = len(best_proposed_solution)
    for i in range(count_assets):
        incremented_solution = copy(current_best_solution)
        incremented_solution[i] = (
            incremented_solution[i] + per_contract_value[i] * direction[i]
        )
        incremented_objective_value = evaluate_tracking_error(
            incremented_solution, data_for_single_period_as_np
        )

        if incremented_objective_value < best_proposed_value:
            best_proposed_value = incremented_objective_value
            best_proposed_solution = incremented_solution

    return best_proposed_value, best_proposed_solution


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

    ## SET CAPITAL TO BE PITIFULLY SMALL SO OPTIMISATION HAS STUFF TO DO
    capital = 50000

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
    rules_spec = [
        dict(function="ewmac", fast_span=16),
        dict(function="ewmac", fast_span=32),
        dict(function="ewmac", fast_span=64),
    ]

    ## FIRST WE SET UP A SYSTEM FOR THE D.O. TO WORK ON
    unrounded_position_contracts_dict = calculate_position_dict_with_forecast_applied(
        adjusted_prices_dict=adjusted_prices_dict,
        carry_prices_dict=carry_prices_dict,
        std_dev_dict=std_dev_dict,
        average_position_contracts_dict=average_position_contracts_dict,
        rule_spec=rules_spec,
    )

    ## positions are generally less than one lot

    perc_return_unrounded_dict = calculate_perc_returns_for_dict_with_costs(
        position_contracts_dict=unrounded_position_contracts_dict,
        fx_series=fx_series_dict,
        multipliers=multipliers,
        capital=capital,
        adjusted_prices=adjusted_prices_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )

    algo_to_use = greedy_algo_across_integer_values
    optimised_positions_dict = dynamically_optimise_positions(
        capital=capital,
        current_prices_dict=current_prices_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        adjusted_prices_dict=adjusted_prices_dict,
        fx_series_dict=fx_series_dict,
        multipliers=multipliers,
        unrounded_position_contracts_dict=unrounded_position_contracts_dict,
        std_dev_dict=std_dev_dict,
        algo_to_use=algo_to_use,
    )

    perc_return_rounded_dict = calculate_perc_returns_for_dict_with_costs(
        position_contracts_dict=optimised_positions_dict,
        fx_series=fx_series_dict,
        multipliers=multipliers,
        capital=capital,
        adjusted_prices=adjusted_prices_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )
