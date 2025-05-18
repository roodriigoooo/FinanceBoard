"""
Fundamental analysis page for the Finance Board application.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.utils.helpers import format_pct, format_num, format_currency, format_with_suffix
from modules.visualizations.charts import (
    create_growth_chart, create_dividend_chart, create_balance_sheet_chart,
    create_earnings_chart, create_income_statement_chart, create_cash_flow_chart,
    create_earnings_estimate_chart
)
from config.settings import COLORS


def render_fundamental_page(ticker, asset_name, info, financials_q, financials_a,
                           balance_sheet_q, balance_sheet_a, cash_flow_q, cash_flow_a,
                           earnings_estimate, recommendations, dividends):
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

                # Balance Sheet Snapshot (Quarterly YoY % Change)
                st.markdown("#### Balance Sheet Snapshot (Quarterly YoY % Change)")

                # Select a few key items and show YoY % change for the latest quarter
                key_bs_items = ['Total Assets', 'Total Liab', 'Total Stockholder Equity', 'Cash']
                bs_snapshot = balance_sheet_q.loc[balance_sheet_q.index.isin(key_bs_items)].T.sort_index(ascending=False)

                if len(bs_snapshot) >= 5:  # Need at least 5 quarters for YoY change (current + 4 previous)
                    bs_yoy_change = bs_snapshot.pct_change(periods=4) * 100  # YoY change
                    latest_q_yoy = bs_yoy_change.iloc[0:1]  # Latest quarter's YoY change

                    if not latest_q_yoy.empty:
                        st.caption(f"Latest Quarter: {latest_q_yoy.index[0].strftime('%Y-%m-%d')}")
                        # Transpose for better display and select only relevant items
                        display_df = latest_q_yoy.T
                        display_df.columns = ["YoY % Change"]
                        st.dataframe(display_df.style.format("{:.2f}%", na_rep="–"))
                    else:
                        st.info("Not enough quarterly data for YoY balance sheet snapshot.")
                else:
                    st.info("Not enough quarterly data for YoY balance sheet snapshot.")
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

    with fund_tabs[5]:  # Analyst Ratings
        st.markdown("### Analyst Recommendations")

        if not recommendations.empty:
            # Display the most recent recommendations
            st.markdown("#### Recent Analyst Recommendations")

            # Format the dataframe for display
            rec_df = recommendations.copy()
            rec_df = rec_df.sort_index(ascending=False).head(10)  # Show last 10 recommendations

            # Display the dataframe
            st.dataframe(rec_df, use_container_width=True)

            # Visualize recommendations
            st.markdown("#### Recommendation Trends")

            try:
                # Count recommendations by grade
                # Check if 'To Grade' column exists, otherwise use the first column
                if rec_df.empty or len(rec_df.columns) == 0:
                    st.warning("Recommendations data is empty or has no columns.")
                    rec_counts = pd.DataFrame(columns=['Recommendation', 'Count'])
                else:
                    grade_column = 'To Grade' if 'To Grade' in rec_df.columns else rec_df.columns[0]
                    rec_counts = rec_df[grade_column].value_counts().reset_index()
                    rec_counts.columns = ['Recommendation', 'Count']
            except Exception as e:
                st.warning(f"Could not process recommendation counts: {e}")
                rec_counts = pd.DataFrame(columns=['Recommendation', 'Count'])

            # Create pie chart if we have data
            if not rec_counts.empty and 'Count' in rec_counts.columns and rec_counts['Count'].sum() > 0:
                fig = px.pie(
                    rec_counts,
                    values='Count',
                    names='Recommendation',
                    title='Current Analyst Recommendations',
                    color_discrete_sequence=px.colors.sequential.Viridis
                )

                # Update layout
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=0, r=0, t=50, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)
            elif not rec_df.empty:
                st.info("Could not create recommendation pie chart due to data format issues.")

            # Recommendation trend over time
            st.markdown("#### Recommendation History")

            try:
                # Group recommendations by date and grade
                rec_trend = rec_df.reset_index()

                # Check if we have a date column
                date_column = None
                if 'Date' in rec_trend.columns:
                    date_column = 'Date'
                elif rec_trend.index.name == 'Date':
                    date_column = rec_trend.index.name
                elif isinstance(rec_trend.index, pd.DatetimeIndex):
                    # If index is already a DatetimeIndex, use it
                    date_column = rec_trend.index.name or 'index'
                elif rec_trend.columns[0] == 'index' and pd.api.types.is_datetime64_any_dtype(rec_trend['index']):
                    # If first column is a datetime after reset_index
                    date_column = 'index'
                else:
                    # Try to find any datetime column
                    for col in rec_trend.columns:
                        if pd.api.types.is_datetime64_any_dtype(rec_trend[col]):
                            date_column = col
                            break

                if date_column is None:
                    st.warning("Could not find a date column in recommendations data.")
                    rec_trend = pd.DataFrame()  # Empty DataFrame to skip the chart
                else:
                    # Check if 'To Grade' column exists, otherwise use the first non-date column
                    grade_columns = [col for col in rec_trend.columns if col != date_column]
                    grade_column = 'To Grade' if 'To Grade' in rec_trend.columns else (grade_columns[0] if grade_columns else None)

                    if grade_column:
                        rec_trend = rec_trend.groupby([pd.Grouper(key=date_column, freq='M'), grade_column]).size().unstack().fillna(0)
                    else:
                        st.warning("Could not find a grade column in recommendations data.")
                        rec_trend = pd.DataFrame()  # Empty DataFrame to skip the chart
            except Exception as e:
                st.warning(f"Could not process recommendation trend data: {e}")
                rec_trend = pd.DataFrame()  # Empty DataFrame to skip the chart

            # Create stacked bar chart if we have data
            if not rec_trend.empty and len(rec_trend.columns) > 0:
                fig = go.Figure()

                for grade in rec_trend.columns:
                    fig.add_trace(go.Bar(
                        x=rec_trend.index,
                        y=rec_trend[grade],
                        name=grade
                    ))

                # Update layout
                fig.update_layout(
                    title="Recommendation History",
                    xaxis_title="Date",
                    yaxis_title="Count",
                    barmode='stack',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=0, r=0, t=50, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)
            elif not rec_df.empty:
                st.info("Could not create recommendation trend chart due to data format issues.")
            # If rec_df is empty, we already show a message above
        else:
            st.info(f"No analyst recommendations available for {ticker}.")

        # Earnings estimates
        st.markdown("### Earnings Estimates")

        if not earnings_estimate.empty:
            # Display earnings estimates
            st.markdown("#### Earnings Estimates")

            # Format the dataframe for display
            est_df = earnings_estimate.copy()

            # Display the dataframe
            st.dataframe(est_df, use_container_width=True)

            # Try to visualize earnings estimates if possible
            try:
                # First, try to use our enhanced earnings estimate comparison chart
                # We need to get the earnings data from the app.py
                earnings = pd.DataFrame()
                if 'earnings' in st.session_state:
                    earnings = st.session_state.earnings

                if not earnings.empty and not est_df.empty:
                    st.markdown("#### Actual vs. Estimated Earnings")
                    earnings_comparison = create_earnings_estimate_chart(earnings, est_df)
                    if earnings_comparison:
                        st.plotly_chart(earnings_comparison, use_container_width=True)

                # Create a simple bar chart of earnings estimates
                if not est_df.empty and 'Earnings Estimate' in est_df.columns:
                    st.markdown("#### Earnings Estimates by Period")
                    # Extract earnings estimates
                    if isinstance(est_df['Earnings Estimate'], pd.Series):
                        est_data = est_df['Earnings Estimate'].reset_index()
                        est_data.columns = ['Period', 'Estimate']

                        # Create bar chart
                        fig = px.bar(
                            est_data,
                            x='Period',
                            y='Estimate',
                            title='Earnings Estimates',
                            color='Estimate',
                            color_continuous_scale=px.colors.sequential.Viridis
                        )

                        # Update layout
                        fig.update_layout(
                            xaxis_title="Period",
                            yaxis_title="Earnings Estimate ($)",
                            margin=dict(l=0, r=0, t=50, b=0)
                        )

                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Could not visualize earnings estimates: {e}")
        else:
            st.info(f"No earnings estimate data available for {ticker}.")
