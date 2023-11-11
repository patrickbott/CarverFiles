"""
This is the provided example python code for Chapter two of the book:
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

from chapter1 import pd_readcsv, BUSINESS_DAYS_IN_YEAR,\
    calculate_perc_returns, calculate_stats, MONTH

def calculate_standard_deviation_for_risk_targeting(adjusted_price: pd.Series,
                                                    current_price: pd.Series):

    daily_price_changes = adjusted_price.diff()
    percentage_changes = daily_price_changes / current_price.shift(1)

    ## Can do the whole series or recent history
    recent_daily_std = percentage_changes.tail(30).std()

    return recent_daily_std*(BUSINESS_DAYS_IN_YEAR**.5)

def calculate_position_series_given_fixed_risk(capital: float,
                                               risk_target_tau: float,
                                               current_price: pd.Series,
                                               fx: pd.Series,
                                               multiplier: float,
                                               instrument_risk_ann_perc: float) -> pd.Series:

    #N = (Capital × τ) ÷ (Multiplier × Price × FX × σ %)
    position_in_contracts =  capital * risk_target_tau / (multiplier * current_price * fx * instrument_risk_ann_perc)

    return position_in_contracts

def calculate_minimum_capital(multiplier: float,
                              price: float,
                              fx: float,
                              instrument_risk_ann_perc: float,
                              risk_target: float,
                              contracts: int = 4):
    # (4 × Multiplier × Price × FX × σ % ) ÷ τ
    minimum_capital= contracts * multiplier * price * fx * instrument_risk_ann_perc / risk_target

    return minimum_capital

if __name__ == '__main__':
    ## Get the file from https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/sp500.csv
    data = pd_readcsv('sp500.csv')
    data = data.dropna()

    adjusted_price = data.adjusted
    current_price = data.underlying
    multiplier = 5
    risk_target_tau= .2
    fx_series = pd.Series(1, index=data.index)  ## FX rate, 1 for USD / USD

    capital= 100000

    instrument_risk = calculate_standard_deviation_for_risk_targeting(adjusted_price=adjusted_price,
                                                                      current_price=current_price)

    position_contracts_held = calculate_position_series_given_fixed_risk(capital=capital,
                                                                         fx=fx_series,
                                                                         instrument_risk_ann_perc=instrument_risk,
                                                                         risk_target_tau=risk_target_tau,
                                                                         multiplier=multiplier,
                                                                         current_price=current_price)

    perc_return = calculate_perc_returns(
        position_contracts_held=position_contracts_held,
        adjusted_price = adjusted_price,
        fx_series=fx_series,
        capital_required=capital,
        multiplier=multiplier
    )

    print(calculate_stats(perc_return))
    print(calculate_stats(perc_return), MONTH)

    print(calculate_minimum_capital(multiplier=multiplier,
                                    risk_target=risk_target_tau,
                                    fx=1,
                                    instrument_risk_ann_perc=instrument_risk,
                                    price = current_price[-1]))
