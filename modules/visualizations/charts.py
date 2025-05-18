"""
Chart creation functions for the Finance Board application.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config.settings import COLORS


def create_candlestick_chart(df, price, volume, metrics, bench_price_col=None, bdf=None, benchmark=None, ticker=None, asset_name=None):
    """
    Create a candlestick chart with technical indicators.

    Args:
        df: DataFrame with OHLC data
        price: Series with price data
        volume: Series with volume data
        metrics: Dictionary with metrics
        bench_price_col: Column name for benchmark price
        bdf: DataFrame with benchmark data
        benchmark: Benchmark ticker symbol
        ticker: Ticker symbol
        asset_name: Asset name

    Returns:
        Plotly figure
    """
    if not df.empty and all(
            col in df.columns for col in ['Open', 'High', 'Low', 'Close']) and not price.empty and not volume.empty:
        # Create figure with subplots: 1 for candlestick, 1 for volume, 1 for RSI, 1 for MACD
        fig_tech = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,  # Reduced spacing
            row_heights=[0.5, 0.15, 0.15, 0.2]  # Adjusted row heights: Price, Volume, RSI, MACD
        )

        # 1. Candlestick Chart
        ma30 = price.rolling(30).mean()  # price is Adj Close or Close
        sd30 = price.rolling(30).std()
        upper_bb, lower_bb = ma30 + 2 * sd30, ma30 - 2 * sd30
        fig_tech.add_trace(go.Candlestick(x=df.index,
                                          open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                          name="Price", increasing_line_color=COLORS["BLUE"],
                                          decreasing_line_color=COLORS["ORANGE"]),
                           row=1, col=1)
        fig_tech.add_trace(
            go.Scatter(x=ma30.index, y=ma30, name="30‑day MA", line=dict(dash="dot", color=COLORS["YELLOW"], width=1)),
            row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=upper_bb.index, y=upper_bb, name="Upper BB",
                                      line=dict(width=0.7, color="rgba(128,128,128,0.7)")), row=1,
                           col=1)  # Grey for BB
        fig_tech.add_trace(go.Scatter(x=lower_bb.index, y=lower_bb, name="Lower BB",
                                      line=dict(width=0.7, color="rgba(128,128,128,0.7)"), fill="tonexty",
                                      fillcolor="rgba(128,128,128,0.1)"), row=1, col=1)

        if bench_price_col and not bdf.empty:
            if not df['Close'].empty and not bdf[bench_price_col].empty:
                bench_norm_tech = bdf[bench_price_col] * (df['Close'].iloc[0] / bdf[bench_price_col].iloc[0])
                fig_tech.add_trace(go.Scatter(x=bdf.index, y=bench_norm_tech, name=benchmark,
                                              line=dict(color=COLORS["NEUTRAL"], dash="dash", width=1)), row=1, col=1)

        # 2. Volume Histogram
        fig_tech.add_trace(go.Bar(x=volume.index, y=volume, name="Volume", marker_color=COLORS["NEUTRAL"]), row=2, col=1)

        # 3. RSI Plot
        rsi_series = metrics.get("rsi_series")
        if rsi_series is not None and not rsi_series.empty:
            fig_tech.add_trace(
                go.Scatter(x=rsi_series.index, y=rsi_series, name="RSI", line=dict(color=COLORS["PURPLE"], width=1)), row=3,
                col=1)
            fig_tech.add_hline(y=70, line_dash="dash", line_color=COLORS["RED"], line_width=0.5, row=3, col=1)
            fig_tech.add_hline(y=30, line_dash="dash", line_color=COLORS["GREEN"], line_width=0.5, row=3, col=1)

        # 4. MACD Plot
        macd_line = metrics.get("macd_line")
        signal_line = metrics.get("signal_line")
        macd_histogram = metrics.get("macd_histogram")

        if macd_line is not None and not macd_line.empty:
            fig_tech.add_trace(
                go.Scatter(x=macd_line.index, y=macd_line, name="MACD Line", line=dict(color=COLORS["BLUE"], width=1)),
                row=4, col=1)
        if signal_line is not None and not signal_line.empty:
            fig_tech.add_trace(go.Scatter(x=signal_line.index, y=signal_line, name="Signal Line",
                                          line=dict(color=COLORS["ORANGE"], width=1)), row=4, col=1)
        if macd_histogram is not None and not macd_histogram.empty:
            colors = [COLORS["BLUE"] if val >= 0 else COLORS["ORANGE"] for val in macd_histogram]
            fig_tech.add_trace(
                go.Bar(x=macd_histogram.index, y=macd_histogram, name="MACD Hist", marker_color=colors), row=4,
                col=1)

        # Create title using ticker and asset_name if provided
        chart_title = "Interactive Candlestick Chart"
        if ticker and asset_name:
            chart_title = f"{asset_name} ({ticker}) - Technical Analysis"
        elif ticker:
            chart_title = f"{ticker} - Technical Analysis"

        fig_tech.update_layout(
            title=chart_title,
            height=800,  # Increased height for subplots
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False,  # Main chart's slider off
            xaxis2_rangeslider_visible=False,
            xaxis3_rangeslider_visible=False,
            xaxis4_rangeslider_visible=True  # Show slider only on the bottom chart (MACD)
        )
        # Update y-axis titles
        fig_tech.update_yaxes(title_text="Price", row=1, col=1)
        fig_tech.update_yaxes(title_text="Volume", row=2, col=1)
        fig_tech.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        fig_tech.update_yaxes(title_text="MACD", row=4, col=1)

        return fig_tech

    return None


def create_growth_chart(financials_a):
    """
    Create a bar chart showing revenue and earnings growth.
    """
    if not financials_a.empty and 'Total Revenue' in financials_a.index and 'Net Income' in financials_a.index:
        revenue = financials_a.loc['Total Revenue'].sort_index()
        earnings = financials_a.loc['Net Income'].sort_index()

        revenue_growth = revenue.pct_change() * 100
        earnings_growth = earnings.pct_change() * 100

        growth_df = pd.DataFrame({
            'Year': revenue_growth.index.year,
            'Revenue Growth (%)': revenue_growth.values,
            'Earnings Growth (%)': earnings_growth.values
        }).dropna()

        if not growth_df.empty:
            fig_growth = px.bar(
                growth_df,
                x='Year',
                y=['Revenue Growth (%)', 'Earnings Growth (%)'],
                barmode='group',
                labels={"value": "Growth Rate (%)", "variable": "Metric"},
                color_discrete_map={'Revenue Growth (%)': COLORS["BLUE"], 'Earnings Growth (%)': COLORS["ORANGE"]}
            )

            # Add a reference line at 0%
            fig_growth.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

            # Update layout
            fig_growth.update_layout(
                title="Annual Growth Rates",
                margin=dict(l=0, r=0, t=50, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )

            # Update y-axis to show percentage format
            fig_growth.update_yaxes(ticksuffix="%")

            return fig_growth

    return None


def create_cash_flow_chart(cash_flow):
    """
    Create an enhanced chart for cash flow visualization.
    """
    if not cash_flow.empty:
        # Identify key cash flow items
        key_items = [
            'Operating Cash Flow',
            'Capital Expenditure',
            'Free Cash Flow',
            'Cash Flow From Continuing Financing Activities',
            'Cash Flow From Continuing Investing Activities'
        ]

        # Filter items that exist in the cash flow statement
        available_items = [item for item in key_items if item in cash_flow.index]

        if available_items:
            # Create figure
            fig = go.Figure()

            # Transpose data for easier plotting
            cf_data = cash_flow.T

            # Add traces for each key item
            for item in available_items:
                fig.add_trace(
                    go.Bar(
                        x=cf_data.index,
                        y=cf_data[item],
                        name=item,
                        hovertemplate=f"{item}: %{{y:$,.0f}}<extra></extra>"
                    )
                )

            # Calculate and add Free Cash Flow if not already present
            if 'Free Cash Flow' not in available_items and 'Operating Cash Flow' in available_items and 'Capital Expenditure' in available_items:
                fcf = cf_data['Operating Cash Flow'] + cf_data['Capital Expenditure']  # CapEx is negative
                fig.add_trace(
                    go.Bar(
                        x=cf_data.index,
                        y=fcf,
                        name='Free Cash Flow (Calculated)',
                        hovertemplate="Free Cash Flow: %{y:$,.0f}<extra></extra>"
                    )
                )

            # Add FCF to Operating CF ratio if both are available
            if 'Free Cash Flow' in available_items and 'Operating Cash Flow' in available_items:
                fcf_ratio = (cf_data['Free Cash Flow'] / cf_data['Operating Cash Flow']) * 100
                fig.add_trace(
                    go.Scatter(
                        x=cf_data.index,
                        y=fcf_ratio,
                        name='FCF/OCF Ratio (%)',
                        yaxis="y2",
                        line=dict(color=COLORS["PURPLE"], width=2),
                        hovertemplate="FCF/OCF Ratio: %{y:.1f}%<extra></extra>"
                    )
                )

                # Update layout for secondary y-axis
                fig.update_layout(
                    yaxis2=dict(
                        title="FCF/OCF Ratio (%)",
                        titlefont=dict(color=COLORS["PURPLE"]),
                        tickfont=dict(color=COLORS["PURPLE"]),
                        overlaying="y",
                        side="right",
                        ticksuffix="%"
                    )
                )

            # Update layout
            fig.update_layout(
                title="Cash Flow Analysis",
                barmode='group',
                xaxis_title="Period",
                yaxis_title="Amount ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=50, b=0),
                hovermode="x unified"
            )

            # Update y-axis to use dollar format
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")

            return fig

    return None


def create_dividend_chart(dividends):
    """
    Create a chart showing dividend history.
    """
    if not dividends.empty:
        # Create a DataFrame with dividends
        div_df = dividends.reset_index()
        div_df.columns = ['Date', 'Dividend']

        # Add year column for grouping
        div_df['Year'] = div_df['Date'].dt.year

        # Calculate annual dividends
        annual_div = div_df.groupby('Year')['Dividend'].sum().reset_index()

        # Create the figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add individual dividends as scatter points
        fig.add_trace(
            go.Scatter(
                x=div_df['Date'],
                y=div_df['Dividend'],
                mode='markers',
                name='Quarterly Dividends',
                marker=dict(color=COLORS["BLUE"], size=8)
            ),
            secondary_y=False
        )

        # Add annual dividends as bars
        fig.add_trace(
            go.Bar(
                x=annual_div['Year'],
                y=annual_div['Dividend'],
                name='Annual Dividends',
                marker_color=COLORS["GREEN"],
                opacity=0.7
            ),
            secondary_y=True
        )

        # Update layout
        fig.update_layout(
            title='Dividend History',
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=50, b=0)
        )

        # Update axes
        fig.update_yaxes(title_text="Quarterly Dividend ($)", secondary_y=False)
        fig.update_yaxes(title_text="Annual Dividend ($)", secondary_y=True)

        return fig

    return None


def create_balance_sheet_chart(balance_sheet):
    """
    Create a chart visualizing key balance sheet items.
    """
    if not balance_sheet.empty:
        # Create a more comprehensive balance sheet visualization
        # First, identify key categories
        assets_items = ['Total Assets', 'Total Current Assets', 'Cash', 'Short Long Term Investments', 'Net Receivables', 'Inventory']
        liabilities_items = ['Total Liab', 'Total Current Liabilities', 'Accounts Payable', 'Long Term Debt']
        equity_items = ['Total Stockholder Equity', 'Common Stock', 'Retained Earnings']

        # Filter items that exist in the balance sheet
        assets_items = [item for item in assets_items if item in balance_sheet.index]
        liabilities_items = [item for item in liabilities_items if item in balance_sheet.index]
        equity_items = [item for item in equity_items if item in balance_sheet.index]

        # Create subplots: 1 for assets, 1 for liabilities & equity
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Assets", "Liabilities & Equity"),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.5]
        )

        # Transpose data for easier plotting
        bs_data = balance_sheet.T

        # Add assets traces
        for item in assets_items:
            if item in balance_sheet.index:
                fig.add_trace(
                    go.Bar(
                        x=bs_data.index,
                        y=bs_data[item],
                        name=item,
                        hovertemplate=f"{item}: %{{y:$,.0f}}<extra></extra>"
                    ),
                    row=1, col=1
                )

        # Add liabilities traces
        for item in liabilities_items:
            if item in balance_sheet.index:
                fig.add_trace(
                    go.Bar(
                        x=bs_data.index,
                        y=bs_data[item],
                        name=item,
                        hovertemplate=f"{item}: %{{y:$,.0f}}<extra></extra>"
                    ),
                    row=2, col=1
                )

        # Add equity traces
        for item in equity_items:
            if item in balance_sheet.index:
                fig.add_trace(
                    go.Bar(
                        x=bs_data.index,
                        y=bs_data[item],
                        name=item,
                        hovertemplate=f"{item}: %{{y:$,.0f}}<extra></extra>"
                    ),
                    row=2, col=1
                )

        # Update layout
        fig.update_layout(
            title='Balance Sheet Breakdown',
            barmode='group',
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=50, b=0),
            hovermode="x unified"
        )

        # Update y-axes to use dollar format
        fig.update_yaxes(title_text="Amount ($)", tickprefix="$", tickformat=",.0f", row=1, col=1)
        fig.update_yaxes(title_text="Amount ($)", tickprefix="$", tickformat=",.0f", row=2, col=1)

        return fig

    return None


def create_income_statement_chart(financials):
    """
    Create an enhanced chart for income statement visualization.
    """
    if not financials.empty and 'Total Revenue' in financials.index and 'Net Income' in financials.index:
        # Extract key metrics
        revenue = financials.loc['Total Revenue'].sort_index()
        net_income = financials.loc['Net Income'].sort_index()

        # Calculate net margin
        net_margin = (net_income / revenue) * 100

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add revenue bars
        fig.add_trace(
            go.Bar(
                x=revenue.index,
                y=revenue,
                name="Revenue",
                marker_color=COLORS["BLUE"],
                hovertemplate="Revenue: %{y:$,.0f}<extra></extra>"
            ),
            secondary_y=False
        )

        # Add net income bars
        fig.add_trace(
            go.Bar(
                x=net_income.index,
                y=net_income,
                name="Net Income",
                marker_color=COLORS["GREEN"],
                hovertemplate="Net Income: %{y:$,.0f}<extra></extra>"
            ),
            secondary_y=False
        )

        # Add net margin line
        fig.add_trace(
            go.Scatter(
                x=net_margin.index,
                y=net_margin,
                name="Net Margin (%)",
                line=dict(color=COLORS["ORANGE"], width=3),
                hovertemplate="Net Margin: %{y:.2f}%<extra></extra>"
            ),
            secondary_y=True
        )

        # Update layout
        fig.update_layout(
            title="Revenue, Net Income & Margin",
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=50, b=0),
            hovermode="x unified"
        )

        # Update axes
        fig.update_yaxes(title_text="Amount ($)", tickprefix="$", tickformat=",.0f", secondary_y=False)
        fig.update_yaxes(title_text="Net Margin (%)", ticksuffix="%", secondary_y=True)

        return fig

    return None


def create_earnings_estimate_chart(earnings, earnings_estimate):
    """
    Create a chart comparing estimated vs. actual earnings.
    """
    if not earnings.empty and not earnings_estimate.empty:
        try:
            # Extract actual earnings
            if isinstance(earnings, pd.DataFrame) and 'Earnings' in earnings.columns:
                actual_earnings = earnings['Earnings'].copy()
            elif isinstance(earnings, pd.DataFrame):
                actual_earnings = earnings.iloc[:, 0].copy()
            else:
                actual_earnings = pd.Series(dtype=float)

            # Extract earnings estimates
            if 'Earnings Estimate' in earnings_estimate.columns:
                estimated_earnings = earnings_estimate['Earnings Estimate'].copy()
            elif len(earnings_estimate.columns) > 0:
                estimated_earnings = earnings_estimate.iloc[:, 0].copy()
            else:
                estimated_earnings = pd.Series(dtype=float)

            # Create figure
            fig = go.Figure()

            # Add actual earnings
            if not actual_earnings.empty:
                fig.add_trace(go.Bar(
                    x=actual_earnings.index,
                    y=actual_earnings,
                    name='Actual Earnings',
                    marker_color=COLORS["BLUE"],
                    hovertemplate="Actual: %{y:.2f}<extra></extra>"
                ))

            # Add estimated earnings
            if not estimated_earnings.empty:
                fig.add_trace(go.Bar(
                    x=estimated_earnings.index,
                    y=estimated_earnings,
                    name='Estimated Earnings',
                    marker_color=COLORS["ORANGE"],
                    opacity=0.7,
                    hovertemplate="Estimate: %{y:.2f}<extra></extra>"
                ))

            # Update layout
            fig.update_layout(
                title='Actual vs. Estimated Earnings',
                barmode='group',
                xaxis_title='Period',
                yaxis_title='Earnings Per Share ($)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=50, b=0),
                hovermode="x unified"
            )

            return fig
        except Exception as e:
            print(f"Error creating earnings estimate chart: {e}")
            return None

    return None


def create_earnings_chart(earnings):
    """
    Create a chart showing earnings history.
    """
    if not earnings.empty:
        # Convert earnings to DataFrame if it's a Series
        if isinstance(earnings, pd.Series):
            earnings = earnings.reset_index()
            earnings.columns = ['Year', 'Earnings']

        # Create the figure
        fig = go.Figure()

        # Add earnings as bars
        fig.add_trace(go.Bar(
            x=earnings.index,
            y=earnings['Earnings'] if 'Earnings' in earnings.columns else earnings.iloc[:, 0],
            name='Earnings',
            marker_color=COLORS["BLUE"],
            hovertemplate="Earnings: %{y:.2f}<extra></extra>"
        ))

        # Add revenue as bars if available
        if 'Revenue' in earnings.columns:
            fig.add_trace(go.Bar(
                x=earnings.index,
                y=earnings['Revenue'],
                name='Revenue',
                marker_color=COLORS["GREEN"],
                hovertemplate="Revenue: %{y:.2f}<extra></extra>"
            ))

        # Update layout
        fig.update_layout(
            title='Earnings History',
            barmode='group',
            xaxis_title='Year',
            yaxis_title='Amount ($)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=50, b=0),
            hovermode="x unified"
        )

        return fig

    return None


def create_performance_comparison_chart(price, bench_price=None, ticker=None, benchmark=None):
    """
    Create a chart comparing the performance of a stock against a benchmark.

    Args:
        price: Series with price data
        bench_price: Series with benchmark price data
        ticker: Ticker symbol
        benchmark: Benchmark ticker symbol
    """
    if price is None or price.empty:
        return None

    # Create a DataFrame for the normalized prices
    df = pd.DataFrame({'Date': price.index, ticker: price.values})
    df = df.set_index('Date')

    # Normalize the price series to start at 100
    df[f'{ticker} (normalized)'] = (price / price.iloc[0]) * 100

    # Add benchmark if available
    if bench_price is not None and not bench_price.empty and len(bench_price) > 1:
        # Ensure the benchmark has the same start date
        aligned_bench = bench_price.reindex(price.index, method='ffill')
        if not aligned_bench.empty:
            df[benchmark] = aligned_bench
            df[f'{benchmark} (normalized)'] = (aligned_bench / aligned_bench.iloc[0]) * 100

    # Create the figure
    fig = go.Figure()

    # Add the normalized price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[f'{ticker} (normalized)'],
        name=ticker,
        line=dict(color=COLORS["BLUE"], width=2),
        hovertemplate=f"{ticker}: %{{y:.2f}}<extra></extra>"
    ))

    # Add the benchmark line if available
    if benchmark and f'{benchmark} (normalized)' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[f'{benchmark} (normalized)'],
            name=benchmark,
            line=dict(color=COLORS["ORANGE"], width=2, dash='dash'),
            hovertemplate=f"{benchmark}: %{{y:.2f}}<extra></extra>"
        ))

    # Add a reference line at 100
    fig.add_hline(y=100, line_dash="dot", line_color="gray", line_width=1)

    # Update layout
    fig.update_layout(
        title='Relative Performance (Base = 100)',
        xaxis_title='Date',
        yaxis_title='Performance Index',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode="x unified"
    )

    return fig
