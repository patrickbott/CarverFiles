"""
This is the provided example python code for Chapter twenty six of the book:
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

from chapter1 import pd_readcsv
from chapter3 import standardDeviation
from chapter4 import (
    create_fx_series_given_adjusted_prices_dict,
    calculate_variable_standard_deviation_for_risk_targeting_from_dict,
    calculate_position_series_given_variable_risk_for_dict,
)
from chapter5 import calculate_perc_returns_with_costs

## GET THE FOLLOWING FILE FROM: https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/correlation_estimate.py
from correlation_estimate import get_row_of_series_before_date


FORECAST_SCALAR = 9.3
AVG_ABS_FORECAST = 10.0


def get_data_dict_with_hourly_adjusted(instrument_list):

    adjusted_prices_hourly_dict = dict(
        [
            (instrument_code, pd_readcsv("%s_hourly.csv" % instrument_code))
            for instrument_code in instrument_list
        ]
    )

    all_data_daily = dict(
        [
            (instrument_code, pd_readcsv("%s.csv" % instrument_code))
            for instrument_code in instrument_list
        ]
    )

    current_prices_daily_dict = dict(
        [
            (instrument_code, data_for_instrument.underlying)
            for instrument_code, data_for_instrument in all_data_daily.items()
        ]
    )

    adjusted_prices_daily_dict = dict(
        [
            (instrument_code, data_for_instrument.adjusted)
            for instrument_code, data_for_instrument in all_data_daily.items()
        ]
    )

    return (
        adjusted_prices_hourly_dict,
        adjusted_prices_daily_dict,
        current_prices_daily_dict,
    )


def generate_pandl_across_instruments_for_hourly_data(
    adjusted_prices_daily_dict: dict,
    current_prices_daily_dict: dict,
    adjusted_prices_hourly_dict: dict,
    std_dev_dict: dict,
    average_position_contracts_dict: dict,
    fx_series_dict: dict,
    multipliers: dict,
    commission_per_contract_dict: dict,
    capital: float,
    tick_size_dict: dict,
    bid_ask_spread_dict: dict,
    trade_calculation_function,
) -> dict:

    list_of_instruments = list(adjusted_prices_hourly_dict.keys())

    pandl_dict = dict(
        [
            (
                instrument_code,
                calculate_pandl_series_for_instrument(
                    adjusted_hourly_prices=adjusted_prices_hourly_dict[instrument_code],
                    current_daily_prices=current_prices_daily_dict[instrument_code],
                    adjusted_daily_prices=adjusted_prices_daily_dict[instrument_code],
                    daily_stdev=std_dev_dict[instrument_code],
                    average_position_daily=average_position_contracts_dict[
                        instrument_code
                    ],
                    fx_series=fx_series_dict[instrument_code],
                    multiplier=multipliers[instrument_code],
                    commission_per_contract=commission_per_contract_dict[
                        instrument_code
                    ],
                    tick_size=tick_size_dict[instrument_code],
                    bid_ask_spread=bid_ask_spread_dict[instrument_code],
                    capital=capital,
                    trade_calculation_function=trade_calculation_function,
                    instrument_code=instrument_code,
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return pandl_dict


def calculate_pandl_series_for_instrument(
    adjusted_daily_prices: pd.Series,
    current_daily_prices: pd.Series,
    adjusted_hourly_prices: pd.Series,
    daily_stdev: pd.Series,
    average_position_daily: pd.Series,
    fx_series: pd.Series,
    multiplier: float,
    capital: float,
    tick_size: float,
    bid_ask_spread: float,
    commission_per_contract: float,
    instrument_code: str,
    trade_calculation_function,
) -> pd.Series:

    list_of_trades = generate_list_of_mr_trades_for_instrument(
        adjusted_daily_prices=adjusted_daily_prices,
        current_daily_prices=current_daily_prices,
        adjusted_hourly_prices=adjusted_hourly_prices,
        daily_stdev=daily_stdev,
        average_position_daily=average_position_daily,
        tick_size=tick_size,
        bid_ask_spread=bid_ask_spread,
        trade_calculation_function=trade_calculation_function,
        instrument_code=instrument_code,
    )

    perc_returns = calculate_perc_returns_from_trade_list(
        list_of_trades=list_of_trades,
        capital=capital,
        fx_series=fx_series,
        commission_per_contract=commission_per_contract,
        current_price_series=current_daily_prices,
        multiplier=multiplier,
        daily_stdev=daily_stdev,
    )

    return perc_returns


def generate_list_of_mr_trades_for_instrument(
    adjusted_daily_prices: pd.Series,
    current_daily_prices: pd.Series,
    adjusted_hourly_prices: pd.Series,
    daily_stdev: pd.Series,
    average_position_daily: pd.Series,
    tick_size: float,
    bid_ask_spread: float,
    instrument_code: str,
    trade_calculation_function,
):

    daily_equilibrium_hourly = calculate_equilibrium(
        adjusted_hourly_prices=adjusted_hourly_prices,
        adjusted_daily_prices=adjusted_daily_prices,
    )

    hourly_stdev_prices = calculate_sigma_p(
        current_daily_prices=current_daily_prices,
        daily_stdev=daily_stdev,
        adjusted_hourly_prices=adjusted_hourly_prices,
    )

    list_of_trades = calculate_trades_for_instrument(
        adjusted_hourly_prices=adjusted_hourly_prices,
        daily_equilibrium_hourly=daily_equilibrium_hourly,
        tick_size=tick_size,
        bid_ask_spread=bid_ask_spread,
        hourly_stdev_prices=hourly_stdev_prices,
        average_position_daily=average_position_daily,
        trade_calculation_function=trade_calculation_function,
        instrument_code=instrument_code,
    )

    return list_of_trades


OrderType = Enum("OrderType", ["LIMIT", "MARKET"])


@dataclass
class Order:
    order_type: OrderType
    qty: int
    limit_price: float = np.nan

    @property
    def is_buy(self):
        return self.qty > 0

    @property
    def is_sell(self):
        return self.qty < 0


class ListOfOrders(list):
    def __init__(self, list_of_orders: List[Order]):
        super().__init__(list_of_orders)

    def drop_buy_limits(self):
        return self.drop_signed_limit_orders(1)

    def drop_sell_limits(self):
        return self.drop_signed_limit_orders(-1)

    def drop_signed_limit_orders(self, order_sign: int):
        new_list = ListOfOrders(
            [
                order
                for order in self
                if true_if_order_is_market_or_order_is_not_of_sign(order, order_sign)
            ]
        )
        return new_list


def true_if_order_is_market_or_order_is_not_of_sign(
    order: Order, order_sign_to_drop: int
):
    if order.order_type == OrderType.MARKET:
        return True

    if not np.sign(order.qty) == order_sign_to_drop:
        return True

    return False


def calculate_trades_for_instrument(
    adjusted_hourly_prices: pd.Series,
    daily_equilibrium_hourly: pd.Series,
    average_position_daily: pd.Series,
    hourly_stdev_prices: pd.Series,
    bid_ask_spread: float,
    tick_size: float,
    instrument_code: str,
    trade_calculation_function,
) -> list:

    list_of_trades = []
    list_of_orders_for_period = ListOfOrders([])
    current_position = 0
    list_of_dates = list(adjusted_hourly_prices.index)

    for relevant_date in list_of_dates[1:]:
        current_price = float(
            get_row_of_series_before_date(
                adjusted_hourly_prices, relevant_date=relevant_date
            )
        )
        if np.isnan(current_price):
            continue

        trade = fill_list_of_orders(
            list_of_orders_for_period,
            current_price=current_price,
            fill_date=relevant_date,
            bid_ask_spread=bid_ask_spread,
        )
        if trade.filled:
            list_of_trades.append(trade)
            current_position = current_position + trade.qty

        current_equilibrium = get_row_of_series_before_date(
            daily_equilibrium_hourly, relevant_date=relevant_date
        )
        current_average_position = get_row_of_series_before_date(
            average_position_daily, relevant_date=relevant_date
        )
        current_hourly_stdev_price = get_row_of_series_before_date(
            hourly_stdev_prices, relevant_date=relevant_date
        )

        list_of_orders_for_period = trade_calculation_function(
            current_position=current_position,
            current_price=current_price,
            current_equilibrium=current_equilibrium,
            current_average_position=current_average_position,
            current_hourly_stdev_price=current_hourly_stdev_price,
            tick_size=tick_size,
            instrument_code=instrument_code,
            relevant_date=relevant_date,
        )

    return list_of_trades


def required_orders_for_mr_system(
    current_position: int,
    current_equilibrium: float,
    current_hourly_stdev_price: float,
    current_price: float,
    current_average_position: float,
    tick_size: float,
    instrument_code: str,
    relevant_date,
) -> ListOfOrders:

    current_forecast = mr_forecast_unclipped(
        current_equilibrium=current_equilibrium,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_price=current_price,
    )

    list_of_orders_for_period = calculate_orders_given_forecast_and_positions(
        current_average_position=current_average_position,
        current_forecast=current_forecast,
        current_equilibrium=current_equilibrium,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_position=current_position,
        tick_size=tick_size,
    )

    if current_forecast < -20:
        list_of_orders_for_period = list_of_orders_for_period.drop_sell_limits()
    elif current_forecast > 20:
        list_of_orders_for_period = list_of_orders_for_period.drop_buy_limits()

    return list_of_orders_for_period


def calculate_orders_given_forecast_and_positions(
    current_forecast: float,
    current_position: int,
    current_equilibrium: float,
    current_hourly_stdev_price: float,
    current_average_position: float,
    tick_size: float,
) -> ListOfOrders:

    current_optimal_position = optimal_position_given_unclipped_forecast(
        current_average_position=current_average_position,
        current_forecast=current_forecast,
    )

    trade_to_optimal = int(np.round(current_optimal_position - current_position))

    if abs(trade_to_optimal) > 1:
        list_of_orders = ListOfOrders(
            [Order(order_type=OrderType.MARKET, qty=trade_to_optimal)]
        )
        return list_of_orders

    buy_limit = get_limit_price_given_resulting_position_with_tick_size_applied(
        current_average_position=current_average_position,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_equilibrium=current_equilibrium,
        tick_size=tick_size,
        number_of_contracts_to_solve_for=current_position + 1,
    )

    sell_limit = get_limit_price_given_resulting_position_with_tick_size_applied(
        current_average_position=current_average_position,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_equilibrium=current_equilibrium,
        tick_size=tick_size,
        number_of_contracts_to_solve_for=current_position - 1,
    )

    return ListOfOrders(
        [
            Order(order_type=OrderType.LIMIT, qty=1, limit_price=buy_limit),
            Order(order_type=OrderType.LIMIT, qty=-1, limit_price=sell_limit),
        ]
    )


def mr_forecast_unclipped(
    current_equilibrium: float,
    current_hourly_stdev_price: float,
    current_price: float,
) -> float:

    raw_forecast = current_equilibrium - current_price
    risk_adjusted_forecast = raw_forecast / current_hourly_stdev_price
    scaled_forecast = risk_adjusted_forecast * FORECAST_SCALAR

    return scaled_forecast


def optimal_position_given_unclipped_forecast(
    current_forecast: float, current_average_position: float
) -> float:

    clipped_forecast = np.clip(current_forecast, -20, 20)

    return clipped_forecast * current_average_position / AVG_ABS_FORECAST


def get_limit_price_given_resulting_position_with_tick_size_applied(
    number_of_contracts_to_solve_for: int,
    current_equilibrium: float,
    current_hourly_stdev_price: float,
    current_average_position: float,
    tick_size: float,
) -> float:

    limit_price = get_limit_price_given_resulting_position(
        number_of_contracts_to_solve_for=number_of_contracts_to_solve_for,
        current_equilibrium=current_equilibrium,
        current_average_position=current_average_position,
        current_hourly_stdev_price=current_hourly_stdev_price,
    )

    return np.round(limit_price / tick_size) * tick_size


def get_limit_price_given_resulting_position(
    number_of_contracts_to_solve_for: int,
    current_equilibrium: float,
    current_hourly_stdev_price: float,
    current_average_position: float,
) -> float:

    return current_equilibrium - (
        number_of_contracts_to_solve_for
        * AVG_ABS_FORECAST
        * current_hourly_stdev_price
        / (FORECAST_SCALAR * current_average_position)
    )


def generate_mr_forecast_series_for_instrument(
    daily_equilibrium_hourly: pd.Series,
    adjusted_hourly_prices: pd.Series,
    hourly_stdev_prices: pd.Series,
) -> pd.Series:

    adjusted_hourly_prices = adjusted_hourly_prices.squeeze()
    raw_forecast = daily_equilibrium_hourly - adjusted_hourly_prices
    risk_adjusted_forecast = raw_forecast / hourly_stdev_prices
    scaled_forecast = risk_adjusted_forecast * FORECAST_SCALAR

    capped_forecast = scaled_forecast.clip(-20, 20)

    return capped_forecast


def calculate_equilibrium(
    adjusted_daily_prices: pd.Series, adjusted_hourly_prices: pd.Series
):

    daily_equilibrium = adjusted_daily_prices.ewm(5).mean()
    daily_equilibrium_hourly = daily_equilibrium.reindex(
        adjusted_hourly_prices.index, method="ffill"
    )

    return daily_equilibrium_hourly


def calculate_sigma_p(
    current_daily_prices: pd.Series,
    adjusted_hourly_prices: pd.Series,
    daily_stdev: pd.Series,
):

    daily_stdev_prices = daily_stdev * current_daily_prices / 16
    hourly_stdev_prices = daily_stdev_prices.reindex(
        adjusted_hourly_prices.index, method="ffill"
    )

    return hourly_stdev_prices


import datetime


@dataclass
class Trade:
    qty: int
    fill_date: datetime.datetime
    current_price: float = np.nan

    @property
    def filled(self):
        return not self.unfilled

    @property
    def unfilled(self):
        return self.qty == 0


not_filled = object()


def fill_list_of_orders(
    list_of_orders: ListOfOrders,
    fill_date: datetime.datetime,
    current_price: float,
    bid_ask_spread: float,
) -> Trade:

    list_of_trades = [
        fill_order(
            order,
            current_price=current_price,
            fill_date=fill_date,
            bid_ask_spread=bid_ask_spread,
        )
        for order in list_of_orders
    ]
    list_of_trades = [trade for trade in list_of_trades if trade.filled]
    if len(list_of_trades) == 0:
        return Trade(qty=0, fill_date=fill_date, current_price=current_price)
    if len(list_of_trades) == 1:
        return list_of_trades[0]

    raise Exception("Impossible for multiple trades to be filled at a given level!")


def fill_order(
    order: Order,
    current_price: float,
    fill_date: datetime.datetime,
    bid_ask_spread: float,
) -> Trade:

    if order.order_type == OrderType.MARKET:
        return fill_market_order(
            order=order,
            current_price=current_price,
            fill_date=fill_date,
            bid_ask_spread=bid_ask_spread,
        )

    elif order.order_type == OrderType.LIMIT:
        return fill_limit_order(
            order=order, fill_date=fill_date, current_price=current_price
        )

    raise Exception("Order type not recognised!")


def fill_market_order(
    order: Order,
    current_price: float,
    fill_date: datetime.datetime,
    bid_ask_spread: float,
) -> Trade:

    if order.is_buy:
        return Trade(
            qty=order.qty,
            fill_date=fill_date,
            current_price=current_price + bid_ask_spread,
        )
    elif order.is_sell:
        return Trade(
            qty=order.qty,
            fill_date=fill_date,
            current_price=current_price - bid_ask_spread,
        )
    else:
        return Trade(qty=0, fill_date=fill_date, current_price=current_price)


def fill_limit_order(
    order: Order, fill_date: datetime.datetime, current_price: float
) -> Trade:

    if order.is_buy:
        if current_price > order.limit_price:
            return Trade(qty=0, fill_date=fill_date, current_price=current_price)

    if order.is_sell:
        if current_price < order.limit_price:
            return Trade(qty=0, fill_date=fill_date, current_price=current_price)

    return Trade(current_price=order.limit_price, qty=order.qty, fill_date=fill_date)


def calculate_perc_returns_from_trade_list(
    list_of_trades: list,
    multiplier: float,
    capital: float,
    fx_series: pd.Series,
    current_price_series: pd.Series,
    commission_per_contract: float,
    daily_stdev: standardDeviation,
) -> pd.Series:

    trade_qty_as_list = [trade.qty for trade in list_of_trades]
    date_index_as_list = [trade.fill_date for trade in list_of_trades]
    price_index_as_list = [trade.current_price for trade in list_of_trades]

    trade_qty_as_series = pd.Series(trade_qty_as_list, index=date_index_as_list)
    trade_prices_as_series = pd.Series(price_index_as_list, index=date_index_as_list)
    position_series = trade_qty_as_series.cumsum()

    perc_returns = calculate_perc_returns_with_costs(
        position_contracts_held=position_series,
        adjusted_price=trade_prices_as_series,
        fx_series=fx_series,
        capital_required=capital,
        multiplier=multiplier,
        cost_per_contract=commission_per_contract,
        stdev_series=daily_stdev,
    )

    return perc_returns


if __name__ == "__main__":
    ## Get the files from:
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/sp500.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/sp500_hourly.csv

    (
        adjusted_prices_hourly_dict,
        adjusted_prices_daily_dict,
        current_prices_daily_dict,
    ) = get_data_dict_with_hourly_adjusted(["sp500"])

    multipliers = dict(sp500=5, us10=1000, us2=2000)
    risk_target_tau = 0.2
    fx_series_dict = create_fx_series_given_adjusted_prices_dict(
        adjusted_prices_daily_dict
    )

    capital = 100000

    idm = 1.4
    instrument_weights = dict(sp500=0.5, us10=0.5, us2=0.3333)

    commission_per_contract_dict = dict(sp500=0.6, us10=1.51, us2=1.51)
    bid_ask_spread_dict = dict(sp500=0.25, us10=1 / 64, us2=(1 / 8) * (1 / 32))
    tick_size_dict = dict(sp500=0.25, us10=1 / 64, us2=(1 / 8) * (1 / 32))

    std_dev_dict = calculate_variable_standard_deviation_for_risk_targeting_from_dict(
        adjusted_prices=adjusted_prices_daily_dict,
        current_prices=current_prices_daily_dict,
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
        trade_calculation_function=required_orders_for_mr_system,
        tick_size_dict=tick_size_dict,
        bid_ask_spread_dict=bid_ask_spread_dict,
    )
