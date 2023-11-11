"""
This is the provided example python code for Chapter one of the book:
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

from enum import Enum
from scipy.stats import norm
import pandas as pd
import numpy as np


DEFAULT_DATE_FORMAT = "%Y-%m-%d"

def pd_readcsv(
    filename: str,
        date_format=DEFAULT_DATE_FORMAT,
        date_index_name: str="index",
) -> pd.DataFrame:

    ans = pd.read_csv(filename)
    ans.index = pd.to_datetime(ans[date_index_name], format=date_format).values

    del ans[date_index_name]

    ans.index.name = None

    return ans


def calculate_perc_returns(position_contracts_held: pd.Series,
                            adjusted_price: pd.Series,
                           fx_series: pd.Series,
                           multiplier: float,
                           capital_required: pd.Series,
                           ) -> pd.Series:

    return_price_points = (adjusted_price - adjusted_price.shift(1))*position_contracts_held.shift(1)

    return_instrument_currency = return_price_points * multiplier
    fx_series_aligned = fx_series.reindex(return_instrument_currency.index, method="ffill")
    return_base_currency = return_instrument_currency * fx_series_aligned

    perc_return = return_base_currency / capital_required

    return perc_return



Frequency = Enum(
    "Frequency",
    "Natural Year Month Week BDay",
)

NATURAL = Frequency.Natural
YEAR = Frequency.Year
MONTH = Frequency.Month
WEEK = Frequency.Week



def calculate_stats(perc_return: pd.Series,
                at_frequency: Frequency = NATURAL) -> dict:

    perc_return_at_freq = sum_at_frequency(perc_return, at_frequency=at_frequency)

    ann_mean = ann_mean_given_frequency(perc_return_at_freq, at_frequency=at_frequency)
    ann_std = ann_std_given_frequency(perc_return_at_freq, at_frequency=at_frequency)
    sharpe_ratio = ann_mean / ann_std

    skew_at_freq = perc_return_at_freq.skew()
    drawdowns = calculate_drawdown(perc_return_at_freq)
    avg_drawdown = drawdowns.mean()
    max_drawdown = drawdowns.max()
    quant_ratio_lower = calculate_quant_ratio_upper(perc_return_at_freq)
    quant_ratio_upper = calculate_quant_ratio_upper(perc_return_at_freq)

    return dict(
        ann_mean = ann_mean,
        ann_std = ann_std,
        sharpe_ratio = sharpe_ratio,
        skew = skew_at_freq,
        avg_drawdown = avg_drawdown,
        max_drawdown = max_drawdown,
        quant_ratio_lower = quant_ratio_lower,
        quant_ratio_upper = quant_ratio_upper
    )

BUSINESS_DAYS_IN_YEAR = 256
WEEKS_PER_YEAR = 52.25
MONTHS_PER_YEAR = 12
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60

PERIODS_PER_YEAR = {
    MONTH: MONTHS_PER_YEAR,
    WEEK: WEEKS_PER_YEAR,
    YEAR: 1

}

def periods_per_year(at_frequency: Frequency):
    if at_frequency == NATURAL:
        return BUSINESS_DAYS_IN_YEAR
    else:
        return PERIODS_PER_YEAR[at_frequency]



def years_in_data(some_data: pd.Series) -> float:
    datediff = some_data.index[-1] - some_data.index[0]
    seconds_in_data = datediff.total_seconds()
    return seconds_in_data / SECONDS_PER_YEAR


def sum_at_frequency(perc_return: pd.Series,
                     at_frequency: Frequency = NATURAL) -> pd.Series:

    if at_frequency == NATURAL:
        return perc_return

    at_frequency_str_dict = {
                        YEAR: "Y",
                        WEEK: "7D",
                        MONTH: "1M"}
    at_frequency_str = at_frequency_str_dict[at_frequency]

    perc_return_at_freq = perc_return.resample(at_frequency_str).sum()

    return perc_return_at_freq


def ann_mean_given_frequency(perc_return_at_freq: pd.Series,
                             at_frequency: Frequency) -> float:

    mean_at_frequency = perc_return_at_freq.mean()
    periods_per_year_for_frequency = periods_per_year(at_frequency)
    annualised_mean = mean_at_frequency * periods_per_year_for_frequency

    return annualised_mean

def ann_std_given_frequency(perc_return_at_freq: pd.Series,
                             at_frequency: Frequency) -> float:

    std_at_frequency = perc_return_at_freq.std()
    periods_per_year_for_frequency = periods_per_year(at_frequency)
    annualised_std = std_at_frequency * (periods_per_year_for_frequency**.5)

    return annualised_std




def calculate_drawdown(perc_return):
    cum_perc_return = perc_return.cumsum()
    max_cum_perc_return = cum_perc_return.rolling(len(perc_return)+1,
                                                  min_periods=1).max()
    return max_cum_perc_return - cum_perc_return

QUANT_PERCENTILE_EXTREME = 0.01
QUANT_PERCENTILE_STD = 0.3
NORMAL_DISTR_RATIO = norm.ppf(QUANT_PERCENTILE_EXTREME) / norm.ppf(QUANT_PERCENTILE_STD)

def calculate_quant_ratio_lower(x):
    x_dm = demeaned_remove_zeros(x)
    raw_ratio = x_dm.quantile(QUANT_PERCENTILE_EXTREME) / x_dm.quantile(
        QUANT_PERCENTILE_STD
    )
    return raw_ratio / NORMAL_DISTR_RATIO

def calculate_quant_ratio_upper(x):
    x_dm = demeaned_remove_zeros(x)
    raw_ratio = x_dm.quantile(1 - QUANT_PERCENTILE_EXTREME) / x_dm.quantile(
        1 - QUANT_PERCENTILE_STD
    )
    return raw_ratio / NORMAL_DISTR_RATIO

def demeaned_remove_zeros(x):
    x[x == 0] = np.nan
    return x - x.mean()


if __name__ == '__main__':
    ## Get the file from https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/sp500.csv
    data = pd_readcsv('sp500.csv')
    data = data.dropna()

    adjusted_price = data.adjusted
    current_price = data.underlying
    multiplier = 5
    fx_series = pd.Series(1, index=data.index)  ## FX rate, 1 for USD / USD
    position_contracts_held = pd.Series(1, index=data.index)  ## applies only to strategy 1

    capital_required = multiplier * current_price  ## applies only to strategy 1

    perc_return = calculate_perc_returns(
        position_contracts_held=position_contracts_held,
        adjusted_price = adjusted_price,
        fx_series=fx_series,
        capital_required=capital_required,
        multiplier=multiplier
    )

    print(calculate_stats(perc_return, at_frequency=MONTH))
