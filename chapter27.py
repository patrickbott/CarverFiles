"""
This is the provided example python code for Chapter twenty seven of the book:
 "Advanced Futures Trading Strategies", by Robert Carver
 https://www.systematicmoney.org/advanced-futures

This code is copyright, Robert Carver 2022.
Shared under https://www.gnu.org/licenses/gpl-3.0.en.html
You may copy, modify, and share this code as long as this header is retained, and you disclose that it has been edited.
This code comes with no warranty, is not guaranteed to be accurate, and the author is not responsible for any losses that may result from it’s use.

Results may not match the book exactly as different data may be used
Results may be different from the corresponding spreadsheet as methods may be slightly different

"""
from enum import Enum
from typing import List
from dataclasses import dataclass
import pandas as pd
import numpy as np
## Next two lines are optional depending on your IDE
import matplotlib

matplotlib.use("TkAgg")

from chapter4 import create_fx_series_given_adjusted_prices_dict, \
    calculate_variable_standard_deviation_for_risk_targeting_from_dict, \
    calculate_position_series_given_variable_risk_for_dict

from chapter7 import calculate_forecast_for_ewmac

from chapter13 import get_attenuation
from chapter26 import generate_pandl_across_instruments_for_hourly_data, get_data_dict_with_hourly_adjusted, \
    calculate_orders_given_forecast_and_positions, ListOfOrders, mr_forecast_unclipped,\
    optimal_position_given_unclipped_forecast, Order, OrderType, AVG_ABS_FORECAST, FORECAST_SCALAR

## GET THE FOLLOWING FILE FROM: https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/correlation_estimate.py
from correlation_estimate import get_row_of_series_before_date

def build_ewmac_storage_dict(adjusted_price_dict: dict,
                             current_prices_daily_dict: dict,
                             std_dev_dict: dict) -> dict:
    list_of_instruments = list(adjusted_price_dict.keys())
    ewmac_dict = dict([
                        (
                           instrument_code,
                            calculate_forecast_for_ewmac(
                                adjusted_price=adjusted_price_dict[instrument_code],
                                 current_price=current_prices_daily_dict[instrument_code],
                                 stdev_ann_perc=std_dev_dict[instrument_code],
                                fast_span=16
                                 )
                        )
                    for instrument_code in list_of_instruments
        ])

    return ewmac_dict

def build_vol_attenuation_dict(
                             std_dev_dict: dict) -> dict:

    list_of_instruments = list(std_dev_dict.keys())
    ewmac_dict = dict([
                        (
                           instrument_code,
                            get_attenuation(std_dev_dict[instrument_code])
                        )
                    for instrument_code in list_of_instruments
        ])

    return ewmac_dict


def required_orders_for_mr_system_with_overlays(current_position: int,
        current_equilibrium: float,
        current_hourly_stdev_price: float,
        current_price: float,
                    current_average_position: float,
                    tick_size: float,
                    instrument_code: str,
                    relevant_date
                    ) -> ListOfOrders:

    current_forecast = mr_forecast_unclipped(current_equilibrium=current_equilibrium,
                                   current_hourly_stdev_price= current_hourly_stdev_price,
                                   current_price=current_price)

    ewmac_sign = current_ewmac_sign(instrument_code, relevant_date)
    if not np.sign(current_forecast)==ewmac_sign:
        current_forecast = 0

    current_atten = current_vol_atten(instrument_code, relevant_date)
    current_forecast = current_forecast * current_atten

    list_of_orders_for_period = calculate_orders_given_forecast_and_positions_and_overlay(
        current_average_position=current_average_position,
        current_forecast=current_forecast,
        current_equilibrium=current_equilibrium,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_position = current_position,
        tick_size=tick_size,
        current_atten = current_atten,
        ewmac_sign = ewmac_sign
    )

    if current_forecast<-20:
        list_of_orders_for_period = list_of_orders_for_period.drop_sell_limits()
    elif current_forecast>20:
        list_of_orders_for_period = list_of_orders_for_period.drop_buy_limits()

    return list_of_orders_for_period

def calculate_orders_given_forecast_and_positions_and_overlay(
        current_forecast: float,
        current_position: int,
        current_equilibrium: float,
        current_hourly_stdev_price: float,
        current_average_position: float,
        tick_size: float,
        ewmac_sign: float,
        current_atten: float
) -> ListOfOrders:

    if not current_position==0:
        if not ewmac_sign==np.sign(current_position):
            list_of_orders = ListOfOrders(
                [
                    Order(order_type=OrderType.MARKET,
                          qty=-current_position)
                ]
            )
            return list_of_orders

    current_optimal_position = optimal_position_given_unclipped_forecast(current_average_position=current_average_position,
                                    current_forecast=current_forecast)

    trade_to_optimal = int(np.round(current_optimal_position - current_position))

    if abs(trade_to_optimal)>1:
        list_of_orders = ListOfOrders(
            [
                Order(order_type=OrderType.MARKET,
                      qty=trade_to_optimal)
            ]
        )
        return list_of_orders


    buy_limit = get_limit_price_given_resulting_position_with_tick_size_applied_for_overlay(
        current_average_position=current_average_position,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_equilibrium=current_equilibrium,
        tick_size=tick_size,
        current_atten=current_atten,

        number_of_contracts_to_solve_for = current_position + 1,

    )

    sell_limit = get_limit_price_given_resulting_position_with_tick_size_applied_for_overlay(
        current_average_position=current_average_position,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_equilibrium=current_equilibrium,
        tick_size=tick_size,
        current_atten = current_atten,

        number_of_contracts_to_solve_for=current_position-1
    )

    return ListOfOrders([
        Order(order_type=OrderType.LIMIT,
              qty=1,
              limit_price=buy_limit),
        Order(order_type=OrderType.LIMIT,
              qty= -1,
              limit_price=sell_limit)
    ])



def current_ewmac_sign(instrument_code: str,
                       relevant_date) -> float:

    return np.sign(get_row_of_series_before_date(ewmac_dict[instrument_code],
                                         relevant_date))


def current_vol_atten(instrument_code: str,
                       relevant_date) -> float:
    return get_row_of_series_before_date(atten_dict[instrument_code],
                                                 relevant_date)

def get_limit_price_given_resulting_position_with_tick_size_applied_for_overlay(
        number_of_contracts_to_solve_for: int,
        current_equilibrium: float,
        current_hourly_stdev_price: float,
        current_average_position: float,
        tick_size: float,
        current_atten: float

)-> float:

    limit_price = \
        get_limit_price_given_resulting_position_with_overlay(
            number_of_contracts_to_solve_for= number_of_contracts_to_solve_for,
        current_equilibrium=current_equilibrium,
        current_average_position=current_average_position,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_atten=current_atten)

    return np.round(limit_price / tick_size) * tick_size

def get_limit_price_given_resulting_position_with_overlay(
        number_of_contracts_to_solve_for: int,
        current_equilibrium: float,
        current_hourly_stdev_price: float,
        current_average_position: float,
        current_atten: float

)-> float:

    return current_equilibrium - (number_of_contracts_to_solve_for *
                                  AVG_ABS_FORECAST *
                                  current_hourly_stdev_price /
                                  (FORECAST_SCALAR * current_atten * current_average_position))



if __name__ == '__main__':
    ## Get the files from:
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/sp500.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/sp500_hourly.csv

    adjusted_prices_hourly_dict, adjusted_prices_daily_dict, current_prices_daily_dict = \
        get_data_dict_with_hourly_adjusted(['sp500'])

    multipliers = dict(sp500=5, us10=1000, us2=2000)
    risk_target_tau = .2
    fx_series_dict = create_fx_series_given_adjusted_prices_dict(adjusted_prices_daily_dict)

    capital = 100000

    idm = 1.4
    instrument_weights = dict(sp500=0.5, us10=.5, us2=.3333)

    commission_per_contract_dict = dict(sp500=.6,
                                        us10=1.51,
                                        us2=1.51)

    bid_ask_spread_dict = dict(sp500=0.25, us10=1 / 64, us2=(1 / 8) * (1 / 32))
    tick_size_dict = dict(sp500=0.25, us10=1 / 64, us2=(1 / 8) * (1 / 32))

    std_dev_dict = calculate_variable_standard_deviation_for_risk_targeting_from_dict(
        adjusted_prices=adjusted_prices_daily_dict,
        current_prices=current_prices_daily_dict
    )

    average_position_contracts_dict = calculate_position_series_given_variable_risk_for_dict(
        capital=capital,
        risk_target_tau=risk_target_tau,
        idm=idm,
        weights=instrument_weights,
        std_dev_dict=std_dev_dict,
        current_prices=current_prices_daily_dict,
        fx_series_dict=fx_series_dict,
        multipliers=multipliers
    )

    ewmac_dict = build_ewmac_storage_dict(adjusted_price_dict=adjusted_prices_daily_dict,
                                          current_prices_daily_dict=current_prices_daily_dict,
                                          std_dev_dict=std_dev_dict)

    atten_dict = build_vol_attenuation_dict(std_dev_dict)

    perc_returns_dict = generate_pandl_across_instruments_for_hourly_data(
        adjusted_prices_hourly_dict=adjusted_prices_hourly_dict,
        adjusted_prices_daily_dict=adjusted_prices_daily_dict,
        fx_series_dict=fx_series_dict,
        multipliers=multipliers,
        commission_per_contract_dict=commission_per_contract_dict,
        average_position_contracts_dict=average_position_contracts_dict,
        std_dev_dict=std_dev_dict,
        current_prices_daily_dict=current_prices_daily_dict,
        capital=capital,
        tick_size_dict=tick_size_dict,
        bid_ask_spread_dict=bid_ask_spread_dict,
        trade_calculation_function=required_orders_for_mr_system_with_overlays

    )
