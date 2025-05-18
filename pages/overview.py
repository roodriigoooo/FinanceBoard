"""
Overview page for the Finance Board application.
"""

import pandas as pd
import streamlit as st
import plotly.express as px

from modules.utils.helpers import format_pct, format_num, format_currency, format_with_suffix, annualised, pick_price_col
from modules.visualizations.charts import create_performance_comparison_chart
from modules.data.loader import fetch_stock_data


def render_overview_page(ticker, asset_name, price, volume, metrics, info, bench_price_col=None, bench_price=None, benchmark=None, bench_total=None, bench_rets=None, rets=None, start_date=None, end_date=None):
    """
    Render the overview page.

    Args:
        ticker: Ticker symbol
        asset_name: Asset name
        price: Series with price data
        volume: Series with volume data
        metrics: Dictionary with metrics
        info: Dictionary with ticker information
        bench_price_col: Column name for benchmark price
        bench_price: Series with benchmark price data
        benchmark: Benchmark ticker symbol
        bench_total: Total benchmark return
        bench_rets: Series with benchmark returns
        rets: Series with stock returns
        start_date: Start date for the analysis period
        end_date: End date for the analysis period
    """
    # Validate inputs to ensure we have the necessary data
    if price is None or price.empty:
        st.error("No price data available. Please try a different ticker or time period.")
        return

    if metrics is None:
        metrics = {}  # Use empty dict to avoid None errors

    if info is None:
        info = {}  # Use empty dict to avoid None errors

    # Now we can safely render the page
    st.header(f"{asset_name} ({ticker}) Overview")

    # --- KPI Cards ---
    st.subheader("Key Performance Indicators")

    # Create a more visually appealing layout with cards
    st.markdown("""
    <style>
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 4px solid #3274a1;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 0.9rem;
        font-weight: 500;
        color: black;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: black;
        margin-bottom: 5px;
    }
    .metric-delta-positive {
        color: #2ca02c;
        font-weight: 500;
        font-size: 1rem;
    }
    .metric-delta-negative {
        color: #d62728;
        font-weight: 500;
        font-size: 1rem;
    }
    .metric-context {
        font-size: 0.8rem;
        color: black;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create a 2x2 grid for the main KPIs
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # KPI 1: Price Change with enhanced context
    price_change_pct_overview = (price.iloc[-1] / price.iloc[-2] - 1) if len(price) >= 2 else None
    price_change_abs = price.iloc[-1] - price.iloc[-2] if len(price) >= 2 else None

    with col1:
        delta_class = "metric-delta-positive" if price_change_pct_overview and price_change_pct_overview > 0 else "metric-delta-negative"
        delta_symbol = "↑" if price_change_pct_overview and price_change_pct_overview > 0 else "↓"
        delta_text = f"{delta_symbol} {format_pct(abs(price_change_pct_overview))}" if price_change_pct_overview is not None else ""

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">DAY CHANGE</div>
            <div class="metric-value">{format_pct(price_change_pct_overview) if price_change_pct_overview is not None else "–"}</div>
            <div class="{delta_class}">{delta_text}</div>
            <div class="metric-context">Absolute: ${f"{price_change_abs:.2f}" if price_change_abs is not None else "0.00"}</div>
        </div>
        """, unsafe_allow_html=True)

    # KPI 2: Market Cap with enhanced context
    market_cap = info.get('marketCap')
    shares_outstanding = info.get('sharesOutstanding')

    with col2:
        market_cap_context = ""
        if market_cap:
            if market_cap > 200e9:
                cap_category = "Mega Cap"
            elif market_cap > 10e9:
                cap_category = "Large Cap"
            elif market_cap > 2e9:
                cap_category = "Mid Cap"
            elif market_cap > 300e6:
                cap_category = "Small Cap"
            else:
                cap_category = "Micro Cap"
            market_cap_context = f"Category: {cap_category}"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MARKET CAP</div>
            <div class="metric-value">{format_with_suffix(market_cap) if market_cap else "–"}</div>
            <div class="metric-context">{market_cap_context}</div>
        </div>
        """, unsafe_allow_html=True)

    # KPI 3: P/E Ratio with enhanced context
    pe_ratio = info.get('trailingPE')
    forward_pe = info.get('forwardPE')
    sector = info.get('sector')

    with col3:
        pe_context = ""
        if pe_ratio and sector:
            # This is simplified - in a real app you'd compare to actual sector averages
            if sector == "Technology" and pe_ratio < 25:
                pe_context = "Below tech sector average"
            elif sector == "Technology" and pe_ratio > 35:
                pe_context = "Above tech sector average"
            elif pe_ratio > 25:
                pe_context = "Above market average"
            else:
                pe_context = "Near market average"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">P/E RATIO (TTM)</div>
            <div class="metric-value">{f"{pe_ratio:.2f}" if pe_ratio else "–"}</div>
            <div class="metric-context">Forward P/E: {f"{forward_pe:.2f}" if forward_pe else "–"}</div>
            <div class="metric-context">{pe_context}</div>
        </div>
        """, unsafe_allow_html=True)

    # KPI 4: Dividend Yield with enhanced context
    dividend_yield = info.get('dividendYield')
    payout_ratio = info.get('payoutRatio')

    with col4:
        dividend_context = ""
        if dividend_yield:
            if dividend_yield > 0.04:
                dividend_context = "High yield stock"
            elif dividend_yield > 0.02:
                dividend_context = "Above average yield"
            elif dividend_yield > 0:
                dividend_context = "Dividend paying stock"
            else:
                dividend_context = "No dividend"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">DIVIDEND YIELD</div>
            <div class="metric-value">{format_pct(dividend_yield) if dividend_yield else "–"}</div>
            <div class="metric-context">Payout Ratio: {format_pct(payout_ratio) if payout_ratio else "–"}</div>
            <div class="metric-context">{dividend_context}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Additional KPIs ---
    st.subheader("Additional Metrics")

    # Create a 2x2 grid for additional metrics
    acol1, acol2 = st.columns(2)
    acol3, acol4 = st.columns(2)

    # 52 Week High/Low combined into one card
    week_high = info.get('fiftyTwoWeekHigh')
    week_low = info.get('fiftyTwoWeekLow')
    current_price = price.iloc[-1] if not price.empty else None

    with acol1:
        if week_high and week_low and current_price:
            # Calculate percentage from 52-week high and low
            pct_from_high = (current_price / week_high - 1) * 100
            pct_from_low = (current_price / week_low - 1) * 100

            # Determine where the current price is in the 52-week range
            range_position = (current_price - week_low) / (week_high - week_low) * 100 if week_high != week_low else 50

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">52-WEEK RANGE</div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px; color: black;">
                    <div style="text-align: left; font-weight: 600; color: black;">Low: ${week_low:.2f}</div>
                    <div style="text-align: right; font-weight: 600; color: black;">High: ${week_high:.2f}</div>
                </div>
                <div style="height: 6px; background-color: #eee; border-radius: 3px; margin-bottom: 10px;">
                    <div style="width: {range_position}%; height: 100%; background-color: #3274a1; border-radius: 3px;"></div>
                </div>
                <div class="metric-context">Current price is {range_position:.1f}% of the 52-week range</div>
                <div class="metric-context">{abs(pct_from_high):.2f}% from 52-week high</div>
                <div class="metric-context">{abs(pct_from_low):.2f}% from 52-week low</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">52-WEEK RANGE</div>
                <div class="metric-value">–</div>
                <div class="metric-context">No data available</div>
            </div>
            """, unsafe_allow_html=True)

    # Volume metrics with context
    avg_volume = info.get('averageVolume')
    current_volume = volume.iloc[-1] if not volume.empty else None

    with acol2:
        volume_context = ""
        if avg_volume and current_volume:
            volume_ratio = current_volume / avg_volume
            if volume_ratio > 2:
                volume_context = "Unusually high volume today"
            elif volume_ratio > 1.5:
                volume_context = "Above average volume today"
            elif volume_ratio < 0.5:
                volume_context = "Below average volume today"
            else:
                volume_context = "Normal trading volume"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">VOLUME METRICS</div>
            <div class="metric-value">{format_num(avg_volume) if avg_volume else "–"}</div>
            <div class="metric-context">Avg. Daily Volume</div>
            <div class="metric-context">Today: {format_num(current_volume) if current_volume else "–"}</div>
            <div class="metric-context">{volume_context}</div>
        </div>
        """, unsafe_allow_html=True)

    # Beta with context
    beta = info.get('beta')

    with acol3:
        beta_context = ""
        if beta:
            if beta > 1.5:
                beta_context = "High volatility compared to market"
            elif beta > 1:
                beta_context = "More volatile than market"
            elif beta > 0.8:
                beta_context = "Similar volatility to market"
            elif beta > 0:
                beta_context = "Less volatile than market"
            else:
                beta_context = "Moves opposite to market"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">BETA</div>
            <div class="metric-value">{f"{beta:.2f}" if beta else "–"}</div>
            <div class="metric-context">{beta_context}</div>
            <div class="metric-context">Beta measures stock's volatility compared to the market</div>
        </div>
        """, unsafe_allow_html=True)

    # Analyst recommendations summary
    target_price = info.get('targetMeanPrice')
    target_high = info.get('targetHighPrice')
    target_low = info.get('targetLowPrice')

    with acol4:
        analyst_context = ""
        if target_price and current_price:
            upside = (target_price / current_price - 1) * 100
            if upside > 20:
                analyst_context = "Strong upside potential"
            elif upside > 5:
                analyst_context = "Moderate upside potential"
            elif upside > -5:
                analyst_context = "Fair valuation"
            else:
                analyst_context = "Potential overvaluation"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ANALYST TARGETS</div>
            <div class="metric-value">${f"{target_price:.2f}" if target_price else "0.00"}</div>
            <div class="metric-context">Mean Target Price</div>
            <div class="metric-context">Range: ${f"{target_low:.2f}" if target_low else "0.00"} - ${f"{target_high:.2f}" if target_high else "0.00"}</div>
            <div class="metric-context">{analyst_context}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Company Profile ---
    st.subheader("Company Profile")

    # Company description
    if 'longBusinessSummary' in info:
        with st.expander("Business Description", expanded=False):
            st.write(info['longBusinessSummary'])

    # Company details in columns
    profile_cols = st.columns(3)

    with profile_cols[0]:
        st.markdown("**General Information**")
        st.markdown(f"**Sector:** {info.get('sector', '–')}")
        st.markdown(f"**Industry:** {info.get('industry', '–')}")
        st.markdown(f"**Website:** [{info.get('website', '–')}]({info.get('website', '#')})")

    with profile_cols[1]:
        st.markdown("**Financial Metrics**")
        st.markdown(f"**Revenue (TTM):** {format_with_suffix(info.get('totalRevenue', None))}")
        st.markdown(f"**Profit Margin:** {format_pct(info.get('profitMargins', None))}")
        st.markdown(f"**Operating Margin:** {format_pct(info.get('operatingMargins', None))}")

    with profile_cols[2]:
        st.markdown("**Valuation Metrics**")
        st.markdown(f"**Forward P/E:** {info.get('forwardPE', '–'):.2f}" if info.get('forwardPE') else "**Forward P/E:** –")
        st.markdown(f"**PEG Ratio:** {info.get('pegRatio', '–'):.2f}" if info.get('pegRatio') else "**PEG Ratio:** –")
        st.markdown(f"**Price/Book:** {info.get('priceToBook', '–'):.2f}" if info.get('priceToBook') else "**Price/Book:** –")

    st.markdown("---")

    # --- Performance Chart ---
    st.subheader("Performance Overview")

    # Create a simple line chart of the stock price
    if not price.empty:
        fig = px.line(
            price,
            title=f"{ticker} Price History",
            labels={"value": "Price ($)", "index": "Date"}
        )
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No price data available for {ticker}.")

    # --- Performance Comparison ---
    st.markdown("---")
    st.subheader("📅 Performance Comparison")

    # Create a visualization of the performance comparison
    if not price.empty:
        # Create the performance comparison chart
        perf_chart = create_performance_comparison_chart(
            price=price,
            bench_price=bench_price if bench_price_col else None,
            ticker=ticker,
            benchmark=benchmark
        )

        if perf_chart:
            st.plotly_chart(perf_chart, use_container_width=True)

    # Create columns for the comparison metrics
    comp_col1, comp_col2, comp_col3 = st.columns(3)

    # Period comparison (if applicable)
    period_days = (end_date - start_date).days if end_date and start_date else None
    if period_days and period_days >= 365:
        prev_df = fetch_stock_data(ticker, start_date - pd.Timedelta(days=period_days), start_date - pd.Timedelta(days=1))
        prev_col = pick_price_col(prev_df)
        if prev_col:
            prev_price = prev_df[prev_col]
            prev_ret = prev_price.iloc[-1] / prev_price.iloc[0] - 1 if len(prev_price) > 1 else None
            if prev_ret is not None:
                period_diff = metrics['total'] - prev_ret
                with comp_col1:
                    st.metric(
                        "Current Period Return",
                        f"{metrics['total']:.2%}",
                        f"{period_diff:.2%}",
                        delta_color="normal"
                    )
                with comp_col2:
                    st.metric(
                        "Previous Period Return",
                        f"{prev_ret:.2%}"
                    )

    # Benchmark comparison (if applicable)
    if bench_price_col and bench_total is not None:
        align = pd.concat([rets, bench_rets], axis=1).dropna()
        beta = None
        alpha = None

        if len(align) > 10:
            beta = align.iloc[:, 0].cov(align.iloc[:, 1]) / align.iloc[:, 1].var()
            alpha = metrics['ann'] - beta * annualised(bench_total, period_days)

        bench_diff = metrics['total'] - bench_total

        with comp_col3:
            st.metric(
                f"{ticker} vs {benchmark}",
                f"{metrics['total']:.2%} vs {bench_total:.2%}",
                f"{bench_diff:.2%}",
                delta_color="normal"
            )

        # Additional metrics in a new row
        if beta is not None or alpha is not None:
            st.markdown("#### Risk-Adjusted Performance Metrics")
            risk_col1, risk_col2, risk_col3 = st.columns(3)

            if beta is not None:
                with risk_col1:
                    st.metric(
                        "Beta",
                        f"{beta:.2f}",
                        help="Beta measures the volatility of a stock relative to the market. A beta > 1 indicates higher volatility than the market."
                    )

            if alpha is not None:
                with risk_col2:
                    st.metric(
                        "Alpha (annualized)",
                        f"{alpha:.2%}",
                        help="Alpha represents the excess return of an investment relative to the return of a benchmark index."
                    )

            # Add Sharpe ratio if available
            if 'sharpe' in metrics and pd.notna(metrics['sharpe']):
                with risk_col3:
                    st.metric(
                        "Sharpe Ratio",
                        f"{metrics['sharpe']:.2f}",
                        help="Sharpe ratio measures the performance of an investment compared to a risk-free asset, after adjusting for its risk."
                    )

    # --- Risk Metrics ---
    st.markdown("---")
    st.subheader("Risk Metrics")

    # Create a 2x2 grid for risk metrics
    rcol1, rcol2 = st.columns(2)
    rcol3, rcol4 = st.columns(2)

    # Volatility with context
    volatility = metrics.get('vol')

    with rcol1:
        vol_context = ""
        if pd.notna(volatility):
            if volatility > 0.4:
                vol_context = "Extremely high volatility"
            elif volatility > 0.3:
                vol_context = "Very high volatility"
            elif volatility > 0.2:
                vol_context = "High volatility"
            elif volatility > 0.15:
                vol_context = "Moderate volatility"
            else:
                vol_context = "Low volatility"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">VOLATILITY (ANNUALIZED)</div>
            <div class="metric-value">{format_pct(volatility) if pd.notna(volatility) else "–"}</div>
            <div class="metric-context">{vol_context}</div>
            <div class="metric-context">Standard deviation of returns, annualized</div>
        </div>
        """, unsafe_allow_html=True)

    # Sharpe Ratio with context
    sharpe = metrics.get('sharpe')

    with rcol2:
        sharpe_context = ""
        if pd.notna(sharpe):
            if sharpe > 1.5:
                sharpe_context = "Excellent risk-adjusted returns"
            elif sharpe > 1:
                sharpe_context = "Good risk-adjusted returns"
            elif sharpe > 0.5:
                sharpe_context = "Average risk-adjusted returns"
            elif sharpe > 0:
                sharpe_context = "Poor risk-adjusted returns"
            else:
                sharpe_context = "Negative risk-adjusted returns"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">SHARPE RATIO</div>
            <div class="metric-value">{f"{sharpe:.2f}" if pd.notna(sharpe) else "–"}</div>
            <div class="metric-context">{sharpe_context}</div>
            <div class="metric-context">Return per unit of risk (higher is better)</div>
        </div>
        """, unsafe_allow_html=True)

    # Max Drawdown with context
    max_drawdown = metrics.get('mdd')

    with rcol3:
        mdd_context = ""
        if pd.notna(max_drawdown):
            if max_drawdown < -0.5:
                mdd_context = "Extreme historical drawdown"
            elif max_drawdown < -0.3:
                mdd_context = "Severe historical drawdown"
            elif max_drawdown < -0.2:
                mdd_context = "Significant historical drawdown"
            elif max_drawdown < -0.1:
                mdd_context = "Moderate historical drawdown"
            else:
                mdd_context = "Minimal historical drawdown"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MAXIMUM DRAWDOWN</div>
            <div class="metric-value">{format_pct(max_drawdown) if pd.notna(max_drawdown) else "–"}</div>
            <div class="metric-context">{mdd_context}</div>
            <div class="metric-context">Largest peak-to-trough decline in the period</div>
        </div>
        """, unsafe_allow_html=True)

    # Value at Risk with context
    var = metrics.get('var')
    cvar = metrics.get('cvar')

    with rcol4:
        var_context = ""
        if pd.notna(var):
            if var < -0.05:
                var_context = "High daily risk"
            elif var < -0.03:
                var_context = "Moderate daily risk"
            elif var < -0.02:
                var_context = "Average daily risk"
            else:
                var_context = "Low daily risk"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">VALUE AT RISK (95%)</div>
            <div class="metric-value">{format_pct(var) if pd.notna(var) else "–"}</div>
            <div class="metric-context">{var_context}</div>
            <div class="metric-context">Expected worst daily loss (95% confidence)</div>
            <div class="metric-context">Conditional VaR: {format_pct(cvar) if pd.notna(cvar) else "–"}</div>
        </div>
        """, unsafe_allow_html=True)
