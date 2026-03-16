from typing import List

import pandas as pd
import yfinance as yf


def _clean_series_to_list(series: pd.Series, years: int) -> List[float]:
    values = pd.to_numeric(series, errors="coerce").dropna().tolist()
    return [float(x) for x in values[:years]]


def get_free_cashflows(ticker: str, years: int = 5) -> List[float]:
    """
    Try to fetch annual Free Cash Flow from yfinance.
    Fallback:
      FCF = Operating Cash Flow - abs(Capital Expenditure)
    if direct Free Cash Flow is unavailable.
    Returns most recent values first.
    """
    stock = yf.Ticker(ticker)
    cashflow_df = stock.cashflow

    if cashflow_df is None or cashflow_df.empty:
        raise ValueError(f"Cashflow data not available for ticker: {ticker}")

    # Preferred path
    for row_name in ["Free Cash Flow", "FreeCashFlow"]:
        if row_name in cashflow_df.index:
            values = _clean_series_to_list(cashflow_df.loc[row_name], years=years)
            if values:
                return values

    # Fallback path
    operating_candidates = ["Operating Cash Flow", "Total Cash From Operating Activities"]
    capex_candidates = ["Capital Expenditure", "Capital Expenditures"]

    operating_name = next((x for x in operating_candidates if x in cashflow_df.index), None)
    capex_name = next((x for x in capex_candidates if x in cashflow_df.index), None)

    if operating_name and capex_name:
        ocf = pd.to_numeric(cashflow_df.loc[operating_name], errors="coerce")
        capex = pd.to_numeric(cashflow_df.loc[capex_name], errors="coerce")

        # CapEx is often negative already; use absolute value to be safe
        fcf = ocf - capex.abs()
        values = fcf.dropna().tolist()[:years]
        values = [float(x) for x in values]
        if values:
            return values

    raise ValueError(
        f"Could not derive Free Cash Flow for ticker: {ticker}. "
        f"Try a different ticker like AAPL, MSFT, GOOGL, AMZN."
    )