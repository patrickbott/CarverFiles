"""
This is the provided example python code for Chapter fifteen of the book:
 "Advanced Futures Trading Strategies", by Robert Carver
 https://www.systematicmoney.org/advanced-futures

It covers seasonal adjustment

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

from chapter3 import standardDeviation
from chapter4 import calculate_variable_standard_deviation_for_risk_targeting_from_dict
from chapter10 import get_data_dict_with_carry, calculate_vol_adjusted_carry


def calculate_seasonally_adjusted_carry(
    original_raw_carry: pd.Series, rolls_per_year: int
) -> pd.Series:

    average_seasonal = calculate_average_seasonal(original_raw_carry)
    shifted_avg_seasonal = calculate_shifted_avg_seasonal(
        average_seasonal=average_seasonal, rolls_per_year=rolls_per_year
    )

    #### STRICTLY SPEAKING THIS IS FORWARD LOOKING...
    average_seasonal_indexed = reindex_seasonal_component_to_index(
        average_seasonal, original_raw_carry.index
    )
    shifted_avg_seasonal_indexed = reindex_seasonal_component_to_index(
        shifted_avg_seasonal, original_raw_carry.index
    )
    net_seasonally_adjusted_carry = original_raw_carry - average_seasonal_indexed
    correctly_seasonally_adjusted_carry = (
        net_seasonally_adjusted_carry + shifted_avg_seasonal_indexed
    )

    return correctly_seasonally_adjusted_carry


def calculate_average_seasonal(original_raw_carry: pd.Series) -> pd.Series:
    original_raw_carry_calendar_days = original_raw_carry.resample("1D").mean()
    original_raw_carry_ffill = original_raw_carry_calendar_days.ffill()
    rolling_average = original_raw_carry_ffill.rolling(365).mean()

    seasonal_component = original_raw_carry_ffill - rolling_average
    seasonal_component_as_matrix = seasonal_matrix(
        seasonal_component, notional_year=NOTIONAL_YEAR
    )
    average_seasonal = seasonal_component_as_matrix.transpose().ewm(5).mean().iloc[-1]

    return average_seasonal


CALENDAR_DAYS_IN_YEAR = 365.25


def calculate_shifted_avg_seasonal(average_seasonal: pd.Series, rolls_per_year: int):
    shift_days = int(CALENDAR_DAYS_IN_YEAR / rolls_per_year)

    shifted_avg_seasonal = shift_seasonal_series(
        average_seasonal, shift_days=shift_days
    )
    return shifted_avg_seasonal


## can have any year but to work both years MUST NOT BE LEAP YEARS
NOTIONAL_YEAR = 2001
NEXT_NOTIONAL_YEAR = NOTIONAL_YEAR + 1


def seasonal_matrix(x, notional_year=NOTIONAL_YEAR):
    ## given some time series x, gives a data frame where each column is a year
    years_to_use = unique_years_in_index(x.index)
    list_of_slices = [
        produce_list_from_x_for_year(x, year, notional_year=notional_year)
        for year in years_to_use
    ]
    concat_list = pd.concat(list_of_slices, axis=1)
    concat_list.columns = years_to_use

    concat_list = concat_list.sort_index()  ## leap years

    return concat_list


def shift_seasonal_series(average_seasonal: pd.Series, shift_days: int):
    ### We want to shift forward, because eg
    ### quarterly roll, holding MARCH; will measure carry MARCH-JUNE
    ### from DEC to MARCH, have MARCH-JUNE carry

    ## get two years
    next_year = NEXT_NOTIONAL_YEAR
    next_year_seasonal = set_year_to_notional_year(
        average_seasonal, notional_year=next_year
    )
    two_years_worth = pd.concat([average_seasonal, next_year_seasonal], axis=0)
    shifted_two_years_worth = two_years_worth.shift(shift_days)
    shifted_average_seasonal_matrix = seasonal_matrix(shifted_two_years_worth)
    shifted_average_seasonal = (
        shifted_average_seasonal_matrix.transpose().ffill().iloc[-1].transpose()
    )

    return shifted_average_seasonal


def reindex_seasonal_component_to_index(seasonal_component, index):
    all_years = unique_years_in_index(index)
    data_with_years = [
        set_year_to_notional_year(seasonal_component, notional_year)
        for notional_year in all_years
    ]
    sequenced_data = pd.concat(data_with_years, axis=0)
    aligned_seasonal = sequenced_data.reindex(index, method="ffill")

    return aligned_seasonal


def unique_years_in_index(index):
    all_years = years_in_index(index)
    unique_years = list(set(all_years))
    unique_years.sort()
    return unique_years


def produce_list_from_x_for_year(x, year, notional_year=NOTIONAL_YEAR):
    list_of_matching_points = index_matches_year(x.index, year)
    matched_x = x[list_of_matching_points]
    matched_x_notional_year = set_year_to_notional_year(
        matched_x, notional_year=notional_year
    )
    return matched_x_notional_year


from copy import copy


def set_year_to_notional_year(x, notional_year=NOTIONAL_YEAR):
    y = copy(x)
    new_index = [
        change_index_day_to_notional_year(index_item, notional_year)
        for index_item in list(x.index)
    ]
    y.index = new_index
    return y


import datetime


def change_index_day_to_notional_year(index_item, notional_year=NOTIONAL_YEAR):
    return datetime.date(notional_year, index_item.month, index_item.day)


def index_matches_year(index, year):

    return [_index_matches_no_leap_days(index_value, year) for index_value in index]


def _index_matches_no_leap_days(index_value, year_to_match):
    if not (index_value.year == year_to_match):
        return False

    if not (index_value.month == 2):
        return True

    if index_value.day == 29:
        return False

    return True


def years_in_index(index):
    index_list = list(index)
    all_years = [item.year for item in index_list]
    return all_years


if __name__ == "__main__":
    ## Get the files from:
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/eurostx.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/eurostx_carry.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/eur_fx.csv
    (
        adjusted_prices_dict,
        current_prices_dict,
        carry_prices_dict,
    ) = get_data_dict_with_carry(["eurostx"])

    std_dev_dict = calculate_variable_standard_deviation_for_risk_targeting_from_dict(
        adjusted_prices=adjusted_prices_dict, current_prices=current_prices_dict
    )

    stdev_ann_perc = std_dev_dict["eurostx"]
    current_price = current_prices_dict["eurostx"]
    carry_price = carry_prices_dict["eurostx"]
    rolls_per_year = 4

    original_raw_carry = calculate_vol_adjusted_carry(
        stdev_ann_perc=stdev_ann_perc, carry_price=carry_price
    )

    seasonally_adj_carry = calculate_seasonally_adjusted_carry(
        original_raw_carry=original_raw_carry, rolls_per_year=rolls_per_year
    )

    ## We would now smooth carry appropriately, apply
    span = 20  ## for example
    smooth_carry = seasonally_adj_carry.ewm(span).mean()
    scaled_carry = smooth_carry * 30
    capped_carry = scaled_carry.clip(-20, 20)
