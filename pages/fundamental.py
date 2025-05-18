"""
Fundamental analysis page for the Finance Board application.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.utils.helpers import format_pct
from modules.visualizations.charts import (
    create_growth_chart, create_dividend_chart, create_balance_sheet_chart,
    create_income_statement_chart, create_cash_flow_chart
)
from config.settings import COLORS


def render_fundamental_page(ticker, asset_name, info, financials_q, financials_a,
                           balance_sheet_q, balance_sheet_a, cash_flow_q, cash_flow_a, dividends):
    """
    Render the fundamental analysis page.

    Args:
        ticker: Ticker symbol
        asset_name: Asset name
        info: Dictionary with ticker information
        financials_q: DataFrame with quarterly financials
        financials_a: DataFrame with annual financials
        balance_sheet_q: DataFrame with quarterly balance sheet
        balance_sheet_a: DataFrame with annual balance sheet
        cash_flow_q: DataFrame with quarterly cash flow
        cash_flow_a: DataFrame with annual cash flow
        earnings: DataFrame with earnings
        earnings_dates: DataFrame with earnings dates
        earnings_estimate: DataFrame with earnings estimates
        recommendations: DataFrame with analyst recommendations
        dividends: Series with dividend history
    """
    st.header(f"{asset_name} ({ticker}) - Fundamental Analysis")

    # Create tabs for different fundamental analysis sections
    fund_tabs = st.tabs(["Key Ratios", "Income Statement", "Balance Sheet", "Cash Flow", "Dividends", "Analyst Ratings"])

    with fund_tabs[0]:  # Key Ratios
        st.markdown("### Key Financial Ratios")

        # --- Key Ratio Grid ---
        if info:
            # First row of ratios
            ratio_cols1 = st.columns(4)
            pe_ratio = info.get('trailingPE', None)
            pb_ratio = info.get('priceToBook', None)
            ps_ratio = info.get('priceToSalesTrailing12Months', None)
            peg_ratio = info.get('pegRatio', None)

            ratio_cols1[0].metric("P/E Ratio (TTM)", f"{pe_ratio:.2f}" if pe_ratio else "–")
            ratio_cols1[1].metric("P/B Ratio", f"{pb_ratio:.2f}" if pb_ratio else "–")
            ratio_cols1[2].metric("P/S Ratio (TTM)", f"{ps_ratio:.2f}" if ps_ratio else "–")
            ratio_cols1[3].metric("PEG Ratio", f"{peg_ratio:.2f}" if peg_ratio else "–")

            # Second row of ratios
            ratio_cols2 = st.columns(4)
            roe = info.get('returnOnEquity', None)
            roa = info.get('returnOnAssets', None)
            profit_margin = info.get('profitMargins', None)
            operating_margin = info.get('operatingMargins', None)

            ratio_cols2[0].metric("Return on Equity", format_pct(roe) if roe else "–")
            ratio_cols2[1].metric("Return on Assets", format_pct(roa) if roa else "–")
            ratio_cols2[2].metric("Profit Margin", format_pct(profit_margin) if profit_margin else "–")
            ratio_cols2[3].metric("Operating Margin", format_pct(operating_margin) if operating_margin else "–")

            # Third row of ratios
            ratio_cols3 = st.columns(4)
            debt_to_equity = info.get('debtToEquity', None)
            current_ratio = info.get('currentRatio', None)
            quick_ratio = info.get('quickRatio', None)
            dividend_yield = info.get('dividendYield', None)

            ratio_cols3[0].metric("Debt to Equity", f"{debt_to_equity:.2f}" if debt_to_equity else "–")
            ratio_cols3[1].metric("Current Ratio", f"{current_ratio:.2f}" if current_ratio else "–")
            ratio_cols3[2].metric("Quick Ratio", f"{quick_ratio:.2f}" if quick_ratio else "–")
            ratio_cols3[3].metric("Dividend Yield", format_pct(dividend_yield) if dividend_yield else "–")
        else:
            st.info("Key ratio data not available.")

        # --- Growth Metrics ---
        st.markdown("### Growth Metrics")

        if not financials_a.empty:
            # Create growth chart
            growth_chart = create_growth_chart(financials_a)
            if growth_chart:
                st.plotly_chart(growth_chart, use_container_width=True)
            else:
                st.info("Not enough annual data for growth chart.")
        else:
            st.info("Annual financial data not available for growth chart.")

    with fund_tabs[1]:  # Income Statement
        st.markdown("### Income Statement")

        # Toggle between quarterly and annual
        income_period = st.radio("Period", ["Quarterly", "Annual"], horizontal=True, key="income_period")

        if income_period == "Quarterly":
            if not financials_q.empty:
                # Format the dataframe for display
                display_df = financials_q.copy()
                # Convert index to strings and format column names as dates
                display_df.index = display_df.index.astype(str)
                display_df.columns = [col.strftime('%Y-%m-%d') for col in display_df.columns]

                # Display the dataframe
                st.dataframe(display_df, use_container_width=True)

                # Visualize key metrics
                if 'Total Revenue' in display_df.index and 'Net Income' in display_df.index:
                    st.markdown("#### Key Income Metrics")

                    # Extract data
                    revenue = financials_q.loc['Total Revenue'].sort_index()
                    net_income = financials_q.loc['Net Income'].sort_index()

                    # Create figure
                    fig = make_subplots(specs=[[{"secondary_y": True}]])

                    # Add revenue bars
                    fig.add_trace(
                        go.Bar(
                            x=revenue.index,
                            y=revenue,
                            name="Revenue",
                            marker_color=COLORS["BLUE"]
                        ),
                        secondary_y=False
                    )

                    # Add net income line
                    fig.add_trace(
                        go.Scatter(
                            x=net_income.index,
                            y=net_income,
                            name="Net Income",
                            line=dict(color=COLORS["GREEN"], width=2)
                        ),
                        secondary_y=True
                    )

                    # Update layout
                    fig.update_layout(
                        title="Quarterly Revenue and Net Income",
                        xaxis_title="Quarter",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=0, r=0, t=50, b=0)
                    )

                    # Update y-axes
                    fig.update_yaxes(title_text="Revenue", secondary_y=False)
                    fig.update_yaxes(title_text="Net Income", secondary_y=True)

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Quarterly income statement data not available.")
        else:  # Annual
            if not financials_a.empty:
                # Format the dataframe for display
                display_df = financials_a.copy()
                # Convert index to strings and format column names as dates
                display_df.index = display_df.index.astype(str)
                display_df.columns = [col.strftime('%Y-%m-%d') for col in display_df.columns]

                # Display the dataframe
                st.dataframe(display_df, use_container_width=True)

                # Visualize key metrics
                if 'Total Revenue' in display_df.index and 'Net Income' in display_df.index:
                    st.markdown("#### Key Income Metrics")

                    # Use our enhanced income statement chart
                    income_chart = create_income_statement_chart(financials_a)
                    if income_chart:
                        st.plotly_chart(income_chart, use_container_width=True)
                    else:
                        # Fallback to the original visualization if our enhanced chart fails
                        # Extract data
                        revenue = financials_a.loc['Total Revenue'].sort_index()
                        net_income = financials_a.loc['Net Income'].sort_index()

                        # Create figure
                        fig = make_subplots(specs=[[{"secondary_y": True}]])

                        # Add revenue bars
                        fig.add_trace(
                            go.Bar(
                                x=revenue.index,
                                y=revenue,
                                name="Revenue",
                                marker_color=COLORS["BLUE"]
                            ),
                            secondary_y=False
                        )

                        # Add net income line
                        fig.add_trace(
                            go.Scatter(
                                x=net_income.index,
                                y=net_income,
                                name="Net Income",
                                line=dict(color=COLORS["GREEN"], width=2)
                            ),
                            secondary_y=True
                        )

                        # Update layout
                        fig.update_layout(
                            title="Annual Revenue and Net Income",
                            xaxis_title="Year",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=0, r=0, t=50, b=0)
                        )

                        # Update y-axes
                        fig.update_yaxes(title_text="Revenue", secondary_y=False)
                        fig.update_yaxes(title_text="Net Income", secondary_y=True)

                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Annual income statement data not available.")

    with fund_tabs[2]:  # Balance Sheet
        st.markdown("### Balance Sheet")

        # Toggle between quarterly and annual
        bs_period = st.radio("Period", ["Quarterly", "Annual"], horizontal=True, key="bs_period")

        if bs_period == "Quarterly":
            if not balance_sheet_q.empty:
                # Format the dataframe for display
                display_df = balance_sheet_q.copy()
                # Convert index to strings and format column names as dates
                display_df.index = display_df.index.astype(str)
                display_df.columns = [col.strftime('%Y-%m-%d') for col in display_df.columns]

                # Display the dataframe
                st.dataframe(display_df, use_container_width=True)

                # Create balance sheet chart
                bs_chart = create_balance_sheet_chart(balance_sheet_q)
                if bs_chart:
                    st.plotly_chart(bs_chart, use_container_width=True)
            else:
                st.info("Quarterly balance sheet data not available.")
        else:  # Annual
            if not balance_sheet_a.empty:
                # Format the dataframe for display
                display_df = balance_sheet_a.copy()
                # Convert index to strings and format column names as dates
                display_df.index = display_df.index.astype(str)
                display_df.columns = [col.strftime('%Y-%m-%d') for col in display_df.columns]

                # Display the dataframe
                st.dataframe(display_df, use_container_width=True)

                # Create balance sheet chart
                bs_chart = create_balance_sheet_chart(balance_sheet_a)
                if bs_chart:
                    st.plotly_chart(bs_chart, use_container_width=True)
            else:
                st.info("Annual balance sheet data not available.")

    with fund_tabs[3]:  # Cash Flow
        st.markdown("### Cash Flow Statement")

        # Toggle between quarterly and annual
        cf_period = st.radio("Period", ["Quarterly", "Annual"], horizontal=True, key="cf_period")

        if cf_period == "Quarterly":
            if not cash_flow_q.empty:
                # Format the dataframe for display
                display_df = cash_flow_q.copy()
                # Convert index to strings and format column names as dates
                display_df.index = display_df.index.astype(str)
                display_df.columns = [col.strftime('%Y-%m-%d') for col in display_df.columns]

                # Display the dataframe
                st.dataframe(display_df, use_container_width=True)

                # Visualize key cash flow metrics
                key_cf_items = ['Operating Cash Flow', 'Capital Expenditure', 'Free Cash Flow']
                if any(item in display_df.index for item in key_cf_items):
                    st.markdown("#### Key Cash Flow Metrics")

                    # Create figure
                    fig = go.Figure()

                    # Add traces for each key item
                    for item in key_cf_items:
                        if item in cash_flow_q.index:
                            fig.add_trace(go.Bar(
                                x=cash_flow_q.columns,
                                y=cash_flow_q.loc[item],
                                name=item
                            ))

                    # Update layout
                    fig.update_layout(
                        title="Quarterly Cash Flow",
                        xaxis_title="Quarter",
                        yaxis_title="Amount ($)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=0, r=0, t=50, b=0)
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Quarterly cash flow data not available.")
        else:  # Annual
            if not cash_flow_a.empty:
                # Format the dataframe for display
                display_df = cash_flow_a.copy()
                # Convert index to strings and format column names as dates
                display_df.index = display_df.index.astype(str)
                display_df.columns = [col.strftime('%Y-%m-%d') for col in display_df.columns]

                # Display the dataframe
                st.dataframe(display_df, use_container_width=True)

                # Visualize key cash flow metrics
                if not cash_flow_a.empty:
                    st.markdown("#### Key Cash Flow Metrics")

                    # Use our enhanced cash flow chart
                    cf_chart = create_cash_flow_chart(cash_flow_a)
                    if cf_chart:
                        st.plotly_chart(cf_chart, use_container_width=True)
                    else:
                        # Fallback to the original visualization if our enhanced chart fails
                        key_cf_items = ['Operating Cash Flow', 'Capital Expenditure', 'Free Cash Flow']
                        if any(item in display_df.index for item in key_cf_items):
                            # Create figure
                            fig = go.Figure()

                            # Add traces for each key item
                            for item in key_cf_items:
                                if item in cash_flow_a.index:
                                    fig.add_trace(go.Bar(
                                        x=cash_flow_a.columns,
                                        y=cash_flow_a.loc[item],
                                        name=item
                                    ))

                            # Update layout
                            fig.update_layout(
                                title="Annual Cash Flow",
                                xaxis_title="Year",
                                yaxis_title="Amount ($)",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                margin=dict(l=0, r=0, t=50, b=0)
                            )

                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Annual cash flow data not available.")

    with fund_tabs[4]:  # Dividends
        st.markdown("### Dividend History")

        if not dividends.empty:
            # Display dividend summary
            div_summary_cols = st.columns(4)

            # Calculate dividend metrics
            current_dividend = dividends.iloc[-1] if not dividends.empty else None
            annual_dividend = dividends.resample('Y').sum().iloc[-1] if not dividends.empty else None
            dividend_yield = info.get('dividendYield', None)
            payout_ratio = info.get('payoutRatio', None)

            div_summary_cols[0].metric("Latest Dividend", f"${current_dividend:.4f}" if current_dividend else "–")
            div_summary_cols[1].metric("Annual Dividend", f"${annual_dividend:.4f}" if annual_dividend else "–")
            div_summary_cols[2].metric("Dividend Yield", format_pct(dividend_yield) if dividend_yield else "–")
            div_summary_cols[3].metric("Payout Ratio", format_pct(payout_ratio) if payout_ratio else "–")

            # Create dividend chart
            div_chart = create_dividend_chart(dividends)
            if div_chart:
                st.plotly_chart(div_chart, use_container_width=True)

            # Display dividend history table
            st.markdown("#### Dividend History Table")

            # Format the dataframe for display
            div_df = dividends.reset_index()
            div_df.columns = ['Date', 'Dividend']
            div_df['Year'] = div_df['Date'].dt.year
            div_df['Quarter'] = div_df['Date'].dt.quarter

            # Sort by date descending
            div_df = div_df.sort_values('Date', ascending=False)

            # Display the dataframe
            st.dataframe(div_df, use_container_width=True)
        else:
            st.info(f"No dividend history available for {ticker}.")

    