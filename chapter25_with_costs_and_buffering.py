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

## GET THE FOLLOWING FILE FROM:  https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/chapter25.py
from chapter25 import *


def greedy_algo_across_integer_values_with_costs_and_buffering(
    data_for_single_period_as_np: dataForSinglePeriodWithWeightsAsNp,
) -> np.array:

    optimised_weights = greedy_algo_across_integer_values_with_costs(
        data_for_single_period_as_np
    )

    previous_weights = data_for_single_period_as_np.previous_position_weights
    covariance = data_for_single_period_as_np.covariance_matrix
    per_contract_value = data_for_single_period_as_np.weight_per_contract

    optimised_weights_with_buffering = calculate_optimised_weights_with_buffering(
        optimised_weights, previous_weights, covariance, per_contract_value
    )

    return optimised_weights_with_buffering


def calculate_optimised_weights_with_buffering(
    optimised_weights: np.array,
    previous_weights: np.array,
    covariance: np.array,
    per_contract_value: np.array,
) -> np.array:

    tracking_error_of_prior = evaluate_tracking_error_for_weights(
        previous_weights, optimised_weights, covariance
    )

    adj_factor = calculate_adjustment_factor_given_tracking_error(
        tracking_error_of_prior=tracking_error_of_prior
    )

    if adj_factor <= 0:
        return previous_weights

    new_optimal_weights_as_np = adjust_weights_with_factor(
        optimised_weights=optimised_weights,
        adj_factor=adj_factor,
        per_contract_value=per_contract_value,
        previous_weights=previous_weights,
    )

    return new_optimal_weights_as_np


def calculate_adjustment_factor_given_tracking_error(
    tracking_error_of_prior: float,
) -> float:

    if tracking_error_of_prior <= 0:
        return 0.0

    tracking_error_buffer = 0.01

    excess_tracking_error = tracking_error_of_prior - tracking_error_buffer

    adj_factor = excess_tracking_error / tracking_error_of_prior
    adj_factor = max(adj_factor, 0.0)

    return adj_factor


def adjust_weights_with_factor(
    optimised_weights: np.array,
    previous_weights: np.array,
    per_contract_value: np.array,
    adj_factor: float,
):

    desired_trades_weight_space = optimised_weights - previous_weights
    adjusted_trades_weight_space = adj_factor * desired_trades_weight_space

    rounded_adjusted_trades_as_weights = (
        calculate_adjusting_trades_rounding_in_contract_space(
            adjusted_trades_weight_space=adjusted_trades_weight_space,
            per_contract_value_as_np=per_contract_value,
        )
    )

    new_optimal_weights = previous_weights + rounded_adjusted_trades_as_weights

    return new_optimal_weights


def calculate_adjusting_trades_rounding_in_contract_space(
    adjusted_trades_weight_space: np.array, per_contract_value_as_np: np.array
) -> np.array:

    adjusted_trades_in_contracts = (
        adjusted_trades_weight_space / per_contract_value_as_np
    )
    rounded_adjusted_trades_in_contracts = np.round(adjusted_trades_in_contracts)
    rounded_adjusted_trades_as_weights = (
        rounded_adjusted_trades_in_contracts * per_contract_value_as_np
    )

    return rounded_adjusted_trades_as_weights


def greedy_algo_across_integer_values_with_costs(
    data_for_single_period_as_np: dataForSinglePeriodWithWeightsAsNp,
) -> np.array:

    ## step 1
    weight_start = data_for_single_period_as_np.starting_weights

    current_best_value = evaluate_with_costs_and_buffering(
        weight_start, data_for_single_period_as_np
    )
    current_best_solution = weight_start

    done = False

    while not done:
        ## step 3 loop
        (
            new_best_proposed_value,
            new_best_proposed_solution,
        ) = find_best_proposed_solution_with_costs_and_buffering(
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


def evaluate_with_costs_and_buffering(
    weights: np.array, data_for_single_period_as_np: dataForSinglePeriodWithWeightsAsNp
) -> float:

    tracking_error = evaluate_tracking_error(weights, data_for_single_period_as_np)
    cost_penalty = calculate_cost_penalty(weights, data_for_single_period_as_np)

    return tracking_error + cost_penalty


def calculate_cost_penalty(
    weights: np.array, data_for_single_period_as_np: dataForSinglePeriodWithWeightsAsNp
) -> float:

    trades_as_weights = weights - data_for_single_period_as_np.previous_position_weights
    cost_of_each_trade = np.abs(
        trades_as_weights * data_for_single_period_as_np.cost_in_weight_terms_as_np
    )

    return 50.0 * np.sum(cost_of_each_trade)


def find_best_proposed_solution_with_costs_and_buffering(
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
        incremented_objective_value = evaluate_with_costs_and_buffering(
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

    algo_to_use = greedy_algo_across_integer_values_with_costs_and_buffering
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
