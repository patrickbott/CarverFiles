from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from chapter3 import standardDeviation
from chapter5 import calculate_deflated_costs

## GET THE FOLLOWING FILE FROM: https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/correlation_estimate.py
from correlation_estimate import (
    covarianceList,
    covarianceEstimate,
    get_common_index,
    calculate_covariance_matrices,
    get_values_for_date_as_dict,
)


class genericValuesPerContract(dict):
    @classmethod
    def allzeros(cls, list_of_keys: list):
        return cls.all_one_value(list_of_keys, value=0.0)

    @classmethod
    def all_one_value(cls, list_of_keys: list, value=0.0):
        return cls.from_weights_and_keys(
            list_of_weights=[value] * len(list_of_keys), list_of_keys=list_of_keys
        )

    @classmethod
    def from_weights_and_keys(cls, list_of_weights: list, list_of_keys: list):
        assert len(list_of_keys) == len(list_of_weights)
        pweights_as_list = [
            (key, weight) for key, weight in zip(list_of_keys, list_of_weights)
        ]

        return cls(pweights_as_list)


class positionContracts(genericValuesPerContract):
    def with_selected_assets_only(self, assets_with_data: list):
        as_dict_with_selected_assets = dict_with_selected_assets_only(
            some_dict=self, assets_with_data=assets_with_data
        )

        return positionContracts(as_dict_with_selected_assets)

    def with_fill_for_missing_assets(
        self, missing_assets: list, fill_value: float = 0.0
    ):
        new_dict = dict([(key, fill_value) for key in missing_assets])

        joint_dict = {**self, **new_dict}

        return positionContracts(joint_dict)


def dict_with_selected_assets_only(some_dict: dict, assets_with_data: list) -> dict:
    return dict([(key, some_dict[key]) for key in assets_with_data])


@dataclass
class dataForOptimisation:
    common_index: list
    list_of_covariance_matrices: covarianceList
    deflated_costs_dict: dict
    unrounded_position_contracts_dict: dict
    fx_series_dict: dict
    multipliers: dict
    capital: float
    current_prices_dict: dict


@dataclass
class dataForSinglePeriod:
    multipliers: dict
    current_prices_this_period: dict
    unrounded_optimal_positions: positionContracts
    fx_rates_this_period: dict
    capital: float
    covariance_this_period: covarianceEstimate
    current_cost_per_contract: dict


class weightPerContract(genericValuesPerContract):
    @classmethod
    def from_single_period_data(cls, data_for_single_period: dataForSinglePeriod):
        notional_exposure_per_contract = (
            notionalExposurePerContract.from_single_period_data(data_for_single_period)
        )
        weight_per_contract_dict = divide_dict_by_float_dict(
            notional_exposure_per_contract, data_for_single_period.capital
        )

        return cls(weight_per_contract_dict)


class notionalExposurePerContract(genericValuesPerContract):
    @classmethod
    def from_single_period_data(cls, data_for_single_period: dataForSinglePeriod):
        ## all keys must match
        multiplier_as_dict_base_fx = multiplied_dict(
            data_for_single_period.fx_rates_this_period,
            data_for_single_period.multipliers,
        )
        notional_exposure_as_dict = multiplied_dict(
            multiplier_as_dict_base_fx,
            data_for_single_period.current_prices_this_period,
        )

        return cls(notional_exposure_as_dict)


class positionWeights(genericValuesPerContract):
    @classmethod
    def from_positions_and_weight_per_contract(
        cls, positions: positionContracts, weights_per_contract: weightPerContract
    ):

        position_weights_as_dict = multiplied_dict(positions, weights_per_contract)

        return cls(position_weights_as_dict)


class costsAsWeights(genericValuesPerContract):
    @classmethod
    def from_costs_capital_and_weight_per_contract(
        cls,
        costs_in_base_currency: dict,
        capital: float,
        weights_per_contract: weightPerContract,
    ):

        costs_as_proportion_of_capital = divide_dict_by_float_dict(
            costs_in_base_currency, capital
        )

        costs_in_weight_terms = divided_dict(
            costs_as_proportion_of_capital, weights_per_contract
        )

        return cls(costs_in_weight_terms)


def position_contracts_from_position_weights(
    position_weights: positionWeights, weights_per_contract: weightPerContract
) -> positionContracts:

    position_contracts_as_dict = divided_dict(position_weights, weights_per_contract)

    return positionContracts(position_contracts_as_dict)


def multiplied_dict(dicta, dictb):
    ## Keys must match
    result_as_dict = dict([(key, value * dictb[key]) for key, value in dicta.items()])

    return result_as_dict


def divided_dict(dicta, dictb):
    ## Keys must match
    result_as_dict = dict([(key, value / dictb[key]) for key, value in dicta.items()])

    return result_as_dict


def divide_dict_by_float_dict(dicta, floatb):
    result_as_dict = dict([(key, value / floatb) for key, value in dicta.items()])

    return result_as_dict


@dataclass
class dataForSinglePeriodWithWeights:
    weight_per_contract: weightPerContract
    previous_position_weights: positionWeights
    unrounded_optimal_position_weights: positionWeights
    covariance_matrix: covarianceEstimate
    costs_in_weight_terms: costsAsWeights

    @classmethod
    def from_data_for_single_period(
        cls,
        previous_position: positionContracts,
        data_for_single_period: dataForSinglePeriod,
    ):
        weight_per_contract = weightPerContract.from_single_period_data(
            data_for_single_period
        )

        previous_position_weights = (
            positionWeights.from_positions_and_weight_per_contract(
                positions=previous_position, weights_per_contract=weight_per_contract
            )
        )

        unrounded_optimal_position_weights = (
            positionWeights.from_positions_and_weight_per_contract(
                positions=data_for_single_period.unrounded_optimal_positions,
                weights_per_contract=weight_per_contract,
            )
        )

        costs_as_weights = costsAsWeights.from_costs_capital_and_weight_per_contract(
            capital=data_for_single_period.capital,
            weights_per_contract=weight_per_contract,
            costs_in_base_currency=data_for_single_period.current_cost_per_contract,
        )

        return cls(
            weight_per_contract,
            previous_position_weights,
            unrounded_optimal_position_weights,
            data_for_single_period.covariance_this_period,
            costs_as_weights,
        )


@dataclass
class dataForSinglePeriodWithWeightsAsNp:
    covariance_matrix: np.array
    unrounded_optimal_position_weights: np.array
    previous_position_weights: np.array
    weight_per_contract: np.array
    starting_weights: np.array
    direction_as_np: np.array
    cost_in_weight_terms_as_np: np.array

    @classmethod
    def from_data_for_single_period_with_weights(
        cls, data_for_single_period_with_weights: dataForSinglePeriodWithWeights
    ):

        unrounded_optimal_position_weights_as_np = np.array(
            list(
                data_for_single_period_with_weights.unrounded_optimal_position_weights.values()
            )
        )
        previous_position_weights_as_np = np.array(
            list(data_for_single_period_with_weights.previous_position_weights.values())
        )
        weight_per_contract_as_np = np.array(
            list(data_for_single_period_with_weights.weight_per_contract.values())
        )
        direction_as_np = np.sign(unrounded_optimal_position_weights_as_np)
        covariance_as_np = data_for_single_period_with_weights.covariance_matrix.values
        starting_weights = zero_np_weights_given_direction_as_np(direction_as_np)
        cost_in_weight_terms_as_np = np.array(
            list(data_for_single_period_with_weights.costs_in_weight_terms.values())
        )

        return cls(
            covariance_as_np,
            unrounded_optimal_position_weights_as_np,
            previous_position_weights_as_np,
            weight_per_contract_as_np,
            starting_weights,
            direction_as_np,
            cost_in_weight_terms_as_np,
        )


def zero_np_weights_given_direction_as_np(direction_as_np: np.array) -> np.array:
    starting_weights = np.array([0.0] * len(direction_as_np))

    return starting_weights


def get_data_for_dynamic_optimisation(
    capital: float,
    fx_series_dict: dict,
    unrounded_position_contracts_dict: dict,
    multipliers: dict,
    std_dev_dict: dict,
    current_prices_dict: dict,
    adjusted_prices_dict: dict,
    cost_per_contract_dict: dict,
) -> dataForOptimisation:

    common_index = get_common_index(unrounded_position_contracts_dict)

    list_of_covariance_matrices = calculate_covariance_matrices(
        adjusted_prices_dict=adjusted_prices_dict,
        current_prices_dict=current_prices_dict,
        std_dev_dict=std_dev_dict,
    )

    deflated_costs_dict = calculate_deflated_costs_dict(
        std_dev_dict=std_dev_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        fx_series_dict=fx_series_dict,
    )

    return dataForOptimisation(
        capital=capital,
        deflated_costs_dict=deflated_costs_dict,
        fx_series_dict=fx_series_dict,
        list_of_covariance_matrices=list_of_covariance_matrices,
        multipliers=multipliers,
        common_index=common_index,
        unrounded_position_contracts_dict=unrounded_position_contracts_dict,
        current_prices_dict=current_prices_dict,
    )


def calculate_deflated_costs_dict(
    cost_per_contract_dict: dict,
    std_dev_dict: dict,
    fx_series_dict: dict,
) -> dict:

    deflated_costs_dict = dict(
        [
            (
                instrument_code,
                calculated_deflated_costs_base_currency(
                    stddev_series=std_dev_dict[instrument_code],
                    cost_per_contract=cost_per_contract_dict[instrument_code],
                    fx_series=fx_series_dict[instrument_code],
                ),
            )
            for instrument_code in list(cost_per_contract_dict.keys())
        ]
    )

    return deflated_costs_dict


def calculated_deflated_costs_base_currency(
    stddev_series: standardDeviation, cost_per_contract: float, fx_series: pd.Series
) -> pd.Series:

    deflated_costs_local = calculate_deflated_costs(
        stddev_series=stddev_series, cost_per_contract=cost_per_contract
    )

    fx_series_aligned = fx_series.reindex(deflated_costs_local.index).ffill()
    deflated_costs_base = deflated_costs_local * fx_series_aligned

    return deflated_costs_base


def get_data_for_relevant_date(
    relevant_date: datetime, data_for_optimisation: dataForOptimisation
) -> dataForSinglePeriod:

    unrounded_optimal_positions = positionContracts(
        get_values_for_date_as_dict(
            relevant_date,
            dict_with_values=data_for_optimisation.unrounded_position_contracts_dict,
        )
    )

    current_prices_this_period = get_values_for_date_as_dict(
        relevant_date, dict_with_values=data_for_optimisation.current_prices_dict
    )

    fx_rates_this_period = get_values_for_date_as_dict(
        relevant_date, dict_with_values=data_for_optimisation.fx_series_dict
    )

    covariance_this_period = data_for_optimisation.list_of_covariance_matrices.most_recent_covariance_before_date(
        relevant_date
    )

    current_cost_per_contract = get_values_for_date_as_dict(
        relevant_date=relevant_date,
        dict_with_values=data_for_optimisation.deflated_costs_dict,
    )

    return dataForSinglePeriod(
        capital=data_for_optimisation.capital,
        covariance_this_period=covariance_this_period,
        current_cost_per_contract=current_cost_per_contract,
        fx_rates_this_period=fx_rates_this_period,
        multipliers=data_for_optimisation.multipliers,
        current_prices_this_period=current_prices_this_period,
        unrounded_optimal_positions=unrounded_optimal_positions,
    )


def which_assets_have_data(data_for_single_period: dataForSinglePeriod) -> list:
    has_cov = data_for_single_period.covariance_this_period.assets_with_data()
    has_price = keys_with_data_in_dict(
        data_for_single_period.current_prices_this_period
    )
    has_fx = keys_with_data_in_dict(data_for_single_period.fx_rates_this_period)
    has_costs = keys_with_data_in_dict(data_for_single_period.current_cost_per_contract)

    return list(
        set(has_fx).intersection(
            set(has_price).intersection(set(has_costs).intersection(has_cov))
        )
    )


def which_assets_without_data(
    data_for_single_period: dataForSinglePeriod, assets_with_data: list
) -> list:

    assets = data_for_single_period.covariance_this_period.columns

    return list(set(assets).difference(set(assets_with_data)))


def keys_with_data_in_dict(some_dict: dict):
    return [key for key, value in some_dict.items() if not np.isnan(value)]


def data_for_single_period_with_valid_assets_only(
    data_for_single_period: dataForSinglePeriod, assets_with_data: list
) -> dataForSinglePeriod:

    ## IMPORTANTLY, AFTER THIS STEP ALL ASSETS WILL BE PROPERLY ALIGNED
    data_for_single_period.covariance_this_period = (
        data_for_single_period.covariance_this_period.subset(assets_with_data)
    )

    data_for_single_period.unrounded_optimal_positions = (
        data_for_single_period.unrounded_optimal_positions.with_selected_assets_only(
            assets_with_data
        )
    )

    ## Rest are just dicts so can use a generic function
    data_for_single_period.fx_rates_this_period = dict_with_selected_assets_only(
        data_for_single_period.fx_rates_this_period, assets_with_data
    )

    data_for_single_period.current_prices_this_period = dict_with_selected_assets_only(
        data_for_single_period.current_prices_this_period, assets_with_data
    )

    data_for_single_period.current_cost_per_contract = dict_with_selected_assets_only(
        data_for_single_period.current_cost_per_contract, assets_with_data
    )

    data_for_single_period.multipliers = dict_with_selected_assets_only(
        data_for_single_period.multipliers, assets_with_data
    )

    return data_for_single_period


def get_initial_positions(position_contracts_dict: dict) -> positionContracts:
    instrument_list = list(position_contracts_dict.keys())
    initial_positions = positionContracts.allzeros(instrument_list)

    return initial_positions


def from_df_to_dict_of_series(position_df: pd.DataFrame) -> dict:
    asset_names = list(position_df.columns)
    result_dict = dict([(key, position_df[key]) for key in asset_names])

    return result_dict
