"""
Main application file for the Finance Board dashboard.
"""

import streamlit as st
import pandas as pd

from modules.ui.components import setup_page_config, render_sidebar, render_top_bar
from modules.data.loader import (
    fetch_stock_data, get_ticker_info,
    get_ticker_financials, get_ticker_earnings,
    get_ticker_recommendations,
    get_ticker_dividends
)
from modules.utils.helpers import pick_price_col, compute_metrics, annualised
from modules.visualizations.charts import create_performance_comparison_chart
from pages.overview import render_overview_page
from pages.technical import render_technical_page
from pages.fundamental import render_fundamental_page
from pages.news import render_news_page


def main():
    """Main application function."""
    # Set up the page
    setup_page_config()

    # Render the sidebar and get user inputs
    ticker, benchmark, start_date, end_date = render_sidebar()

    # Store ticker in session state for other components to access
    st.session_state.ticker = ticker

    # Create a container for loading status messages
    loading_container = st.empty()

    # Use a single spinner for all data loading operations
    with st.spinner("Loading data..."):
        loading_container.info("Fetching price data...")

        # Fetch price data
        df = fetch_stock_data(ticker, start_date, end_date)

        # Validate price data
        price_col = pick_price_col(df)
        if not price_col:
            st.error(f"No price series (Adj Close / Close) found for {ticker}.")
            st.stop()

        price = df[price_col].dropna()
        if price.empty:
            st.error(f"No valid price data available for {ticker}.")
            st.stop()

        # Get volume data
        volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(index=price.index, dtype=float)

        # Calculate metrics
        loading_container.info("Calculating performance metrics...")
        metrics = compute_metrics(price, volume)
        rets = metrics.pop("rets")  # daily returns series for charts/tables

        # Fetch benchmark data if provided
        if benchmark:
            loading_container.info(f"Fetching benchmark data for {benchmark}...")
            bdf = fetch_stock_data(benchmark, start_date, end_date)
        else:
            bdf = pd.DataFrame()

        # Process benchmark data
        bench_price_col = pick_price_col(bdf)
        bench_rets = pd.Series(dtype=float)
        bench_total = None
        bench_price = None

        if bench_price_col:
            bench_price = bdf[bench_price_col].dropna()
            if not bench_price.empty and len(bench_price) > 1:
                bench_rets = bench_price.pct_change().rename(f"{benchmark} Return").dropna()
                bench_total = bench_price.iloc[-1] / bench_price.iloc[0] - 1

        # Fetch company information
        loading_container.info("Fetching company information...")
        info = get_ticker_info(ticker)

        # Validate company information
        if not info:
            loading_container.warning(f"Limited or no company information available for {ticker}.")
            info = {}  # Use empty dict to avoid None errors

        # Get asset name from info or fallback to ticker
        asset_name = ticker
        try:
            short_name = info.get('shortName')
            if short_name:
                asset_name = short_name
        except Exception:
            pass  # Fallback to ticker symbol if name fetch fails

        # Fetch additional data
        loading_container.info("Fetching financial data...")

        # Get financial data
        financials_q, financials_a, balance_sheet_q, balance_sheet_a, cash_flow_q, cash_flow_a = get_ticker_financials(ticker)

        # Get earnings data
        earnings, earnings_dates, earnings_estimate = get_ticker_earnings(ticker)

        # Store earnings data in session state for use in the fundamental page
        st.session_state.earnings = earnings

        # Get recommendations
        recommendations = get_ticker_recommendations(ticker)

        # Get dividends
        dividends = get_ticker_dividends(ticker)

        # Clear the loading message
        loading_container.empty()

    # Render the top bar
    render_top_bar(ticker, asset_name, price, volume)

    # --- Tabbed View ---
    tab_titles = []
    if st.session_state.get("tech_tab_toggle", True):
        tab_titles.append("📊 Overview")
        tab_titles.append("📈 Technical Analysis")
    if st.session_state.get("fund_tab_toggle", True):
        tab_titles.append("📈 Fundamental Analysis")
    if st.session_state.get("news_tab_toggle", True):
        tab_titles.append("📰 News")

    if not tab_titles:
        st.warning("Please select at least one analysis tab from the 'Analysis Toggles' in the sidebar.")
        st.stop()

    tabs = st.tabs(tab_titles)

    # Render the tabs
    if "📊 Overview" in tab_titles:
        with tabs[tab_titles.index("📊 Overview")]:
            render_overview_page(ticker, asset_name, price, volume, metrics, info,
                               bench_price_col=bench_price_col, bench_price=bench_price,
                               benchmark=benchmark, bench_total=bench_total,
                               bench_rets=bench_rets, rets=rets,
                               start_date=start_date, end_date=end_date)

    if "📈 Technical Analysis" in tab_titles:
        with tabs[tab_titles.index("📈 Technical Analysis")]:
            render_technical_page(ticker, asset_name, df, price, volume, metrics, bench_price_col, bdf, benchmark)

    if "📈 Fundamental Analysis" in tab_titles:
        with tabs[tab_titles.index("📈 Fundamental Analysis")]:
            render_fundamental_page(ticker, asset_name, info, financials_q, financials_a,
                                   balance_sheet_q, balance_sheet_a, cash_flow_q, cash_flow_a, dividends)

    if "📰 News" in tab_titles:
        with tabs[tab_titles.index("📰 News")]:
            render_news_page(ticker, asset_name)

    # Performance Comparison section has been moved to the Overview page

    # --- Footer ---
    st.markdown("""
    <style>
    .footer-container {
        margin-top: 30px;
        padding: 15px;
        text-align: center;
        color: #777;
        font-size: 0.8rem;
        border-top: 1px solid #eee;
    }
    </style>
    <div class="footer-container">
        Preset ranges, salient summary & progressive disclosure = <span style="font-style: italic;">simpler, bias‑aware UX</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
