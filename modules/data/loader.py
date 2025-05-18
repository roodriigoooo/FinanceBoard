"""
Data loading functions for the Finance Board application.
"""

from datetime import date, timedelta
import pandas as pd
import streamlit as st
import yfinance as yf
import time
import hashlib

from modules.utils.helpers import flatten_columns, pick_price_col


# Create a cache key based on ticker and dates
def _create_cache_key(ticker, start=None, end=None, suffix=""):
    """Create a unique cache key for the given parameters."""
    key_parts = [ticker]
    if start:
        key_parts.append(str(start))
    if end:
        key_parts.append(str(end))
    if suffix:
        key_parts.append(suffix)

    # Create a hash of the key parts to ensure it's a valid key
    key = hashlib.md5("_".join(key_parts).encode()).hexdigest()
    return f"cache_{key}"


# Function to ensure ticker_cache exists in session state
def ensure_ticker_cache_exists():
    """Ensure that ticker_cache exists in session state."""
    if 'ticker_cache' not in st.session_state:
        st.session_state.ticker_cache = {}


@st.cache_data(ttl=7200, show_spinner=False)  # Cache for 1 hour
def fetch_stock_data(tkr: str, start: date, end: date) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance and clean column index.
    all getters are cached for 2 hours to enhance performance and speed.
    """
    if not tkr:
        return pd.DataFrame()

    # Ensure ticker_cache exists in session state
    ensure_ticker_cache_exists()

    # Check if data is in session state cache
    cache_key = _create_cache_key(tkr, start, end, "stock_data")
    if cache_key in st.session_state.ticker_cache:
        return st.session_state.ticker_cache[cache_key]

    try:
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)
        data = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=False)
        result = flatten_columns(data)

        # Ensure ticker_cache exists and store in session state cache
        ensure_ticker_cache_exists()
        st.session_state.ticker_cache[cache_key] = result
        return result
    except Exception as exc:
        st.error(f"Yahoo download error for {tkr}: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=7200, show_spinner=False)  # Cache for 1 hour
def get_ticker_info(ticker: str) -> dict:
    """
    Get detailed information about a ticker.
    """
    if not ticker:
        return {}

    # Ensure ticker_cache exists in session state
    ensure_ticker_cache_exists()

    # Check if data is in session state cache
    cache_key = _create_cache_key(ticker, suffix="info")
    if cache_key in st.session_state.ticker_cache:
        return st.session_state.ticker_cache[cache_key]

    try:
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)
        ticker_obj = yf.Ticker(ticker)
        result = ticker_obj.info

        # Ensure ticker_cache exists and store in session state cache
        ensure_ticker_cache_exists()
        st.session_state.ticker_cache[cache_key] = result
        return result
    except Exception as e:
        st.warning(f"Could not fetch detailed info for {ticker}: {e}")
        return {}


@st.cache_data(ttl=7200, show_spinner=False)  # Cache for 24 hours
def get_ticker_financials(ticker: str) -> tuple:
    """
    Get financial data for a ticker.
    """
    if not ticker:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    # Ensure ticker_cache exists in session state
    ensure_ticker_cache_exists()

    # Check if data is in session state cache
    cache_key = _create_cache_key(ticker, suffix="financials")
    if cache_key in st.session_state.ticker_cache:
        return st.session_state.ticker_cache[cache_key]

    try:
        # Add a small delay to avoid rate limiting
        time.sleep(0.2)  # Slightly longer delay for financials as it's a heavier call
        ticker_obj = yf.Ticker(ticker)
        result = (
            ticker_obj.quarterly_financials,
            ticker_obj.financials,
            ticker_obj.quarterly_balance_sheet,
            ticker_obj.balance_sheet,
            ticker_obj.quarterly_cashflow,
            ticker_obj.cashflow
        )

        # Ensure ticker_cache exists and store in session state cache
        ensure_ticker_cache_exists()
        st.session_state.ticker_cache[cache_key] = result
        return result
    except Exception as e:
        st.warning(f"Could not fetch financial data for {ticker}: {e}")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


@st.cache_data(ttl=7200, show_spinner=False)  # Cache for 24 hours
def get_ticker_earnings(ticker: str) -> tuple:
    """
    Get earnings data for a ticker.
    """
    if not ticker:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    # Ensure ticker_cache exists in session state
    ensure_ticker_cache_exists()

    # Check if data is in session state cache
    cache_key = _create_cache_key(ticker, suffix="earnings")
    if cache_key in st.session_state.ticker_cache:
        return st.session_state.ticker_cache[cache_key]

    try:
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)
        ticker_obj = yf.Ticker(ticker)
        earnings = ticker_obj.earnings

        # Try to get earnings_dates, but handle if not available
        try:
            earnings_dates = ticker_obj.earnings_dates
        except:
            earnings_dates = pd.DataFrame()

        # Try to get earnings_estimate, but handle if not available
        try:
            earnings_estimate = ticker_obj.earnings_estimate
        except:
            earnings_estimate = pd.DataFrame()

        result = (earnings, earnings_dates, earnings_estimate)

        # Ensure ticker_cache exists and store in session state cache
        ensure_ticker_cache_exists()
        st.session_state.ticker_cache[cache_key] = result
        return result
    except Exception as e:
        st.warning(f"Could not fetch earnings data for {ticker}: {e}")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


@st.cache_data(ttl=7200, show_spinner=False)  # Cache for 24 hours
def get_ticker_recommendations(ticker: str) -> pd.DataFrame:
    """
    Get analyst recommendations for a ticker.
    """
    if not ticker:
        return pd.DataFrame()

    # Ensure ticker_cache exists in session state
    ensure_ticker_cache_exists()

    # Check if data is in session state cache
    cache_key = _create_cache_key(ticker, suffix="recommendations")
    if cache_key in st.session_state.ticker_cache:
        return st.session_state.ticker_cache[cache_key]

    try:
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)
        ticker_obj = yf.Ticker(ticker)
        result = ticker_obj.recommendations

        # Ensure ticker_cache exists and store in session state cache
        ensure_ticker_cache_exists()
        st.session_state.ticker_cache[cache_key] = result
        return result
    except Exception as e:
        st.warning(f"Could not fetch recommendations for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=7200, show_spinner=False)  # Cache for 24 hours
def get_ticker_dividends(ticker: str) -> pd.Series:
    """
    Get dividend history for a ticker.
    """
    if not ticker:
        return pd.Series(dtype=float)

    # Ensure ticker_cache exists in session state
    ensure_ticker_cache_exists()

    # Check if data is in session state cache
    cache_key = _create_cache_key(ticker, suffix="dividends")
    if cache_key in st.session_state.ticker_cache:
        return st.session_state.ticker_cache[cache_key]

    try:
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)
        ticker_obj = yf.Ticker(ticker)
        result = ticker_obj.dividends

        # Ensure ticker_cache exists and store in session state cache
        ensure_ticker_cache_exists()
        st.session_state.ticker_cache[cache_key] = result
        return result
    except Exception as e:
        st.warning(f"Could not fetch dividend data for {ticker}: {e}")
        return pd.Series(dtype=float)
