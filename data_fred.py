import os
from typing import Dict, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred_series(series_id: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch a FRED series as a DataFrame with columns:
    - date (datetime64)
    - value (float)
    """
    fred_key = api_key or os.getenv("FRED_API_KEY")
    if not fred_key:
        raise RuntimeError("Missing FRED_API_KEY in .env")

    params = {
        "series_id": series_id,
        "api_key": fred_key,
        "file_type": "json",
        "sort_order": "asc",
    }

    response = requests.get(FRED_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    observations = payload.get("observations", [])
    if not observations:
        raise ValueError(f"No observations returned for series {series_id}")

    df = pd.DataFrame(observations)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"Series {series_id} became empty after numeric cleaning")

    return df


def latest_value(series_df: pd.DataFrame) -> float:
    if series_df.empty:
        raise ValueError("Series DataFrame is empty")
    return float(series_df["value"].iloc[-1])


def yoy_growth(series_df: pd.DataFrame, periods: int) -> float:
    """
    Year-over-year growth:
    (latest / value_periods_ago) - 1
    """
    if len(series_df) < periods + 1:
        raise ValueError("Not enough observations for YoY growth calculation")

    latest = float(series_df["value"].iloc[-1])
    prev = float(series_df["value"].iloc[-1 - periods])

    if prev == 0:
        raise ValueError("Previous value is zero; cannot compute YoY growth")

    return (latest / prev) - 1.0


def detect_macro_regime(inflation_yoy: float, unemployment: float, gdp_yoy: float) -> str:
    """
    Simple regime detector used for notes/logging and stability interpretation.
    """
    stress = 0

    if inflation_yoy > 0.04:
        stress += 1
    if unemployment > 0.05:
        stress += 1
    if gdp_yoy < 0.01:
        stress += 1

    if stress == 0:
        return "stable"
    if stress == 1:
        return "moderate_stress"
    return "high_stress"


def get_macro_and_risk_free() -> Dict[str, float]:
    """
    Pull the exact fields the project needs:
    - inflation_yoy from CPIAUCSL
    - unemployment from UNRATE
    - gdp_yoy from GDPC1
    - risk_free_10y from DGS10
    """
    cpi_df = fetch_fred_series("CPIAUCSL")
    unrate_df = fetch_fred_series("UNRATE")
    gdp_df = fetch_fred_series("GDPC1")
    dgs10_df = fetch_fred_series("DGS10")

    inflation_yoy = yoy_growth(cpi_df, periods=12)       # CPI monthly YoY
    unemployment = latest_value(unrate_df) / 100.0       # % -> decimal
    gdp_yoy = yoy_growth(gdp_df, periods=4)              # GDP quarterly YoY
    risk_free_10y = latest_value(dgs10_df) / 100.0       # % -> decimal

    regime = detect_macro_regime(inflation_yoy, unemployment, gdp_yoy)

    return {
        "inflation_yoy": float(inflation_yoy),
        "unemployment": float(unemployment),
        "gdp_yoy": float(gdp_yoy),
        "risk_free_10y": float(risk_free_10y),
        "regime": regime,
    }