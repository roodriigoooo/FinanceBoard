"""
Overview page for the Finance Board application.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from modules.utils.helpers import format_pct, format_num, format_currency, format_with_suffix, annualised, pick_price_col
from modules.visualizations.charts import create_performance_comparison_chart
from modules.data.loader import fetch_stock_data
from config.settings import COLORS


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

    # Create a clean, simplified layout with cards that incorporate color-coded trend indicators
    st.markdown(f"""
    <style>
    /* Base card styles */
    .metric-card {{
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 4px solid {COLORS["BLUE"]};
    }}

    /* Card title */
    .metric-title {{
        font-size: 0.9rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 10px;
        letter-spacing: 0.5px;
    }}

    /* Primary metric value */
    .metric-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: #111;
        margin-bottom: 8px;
    }}

    /* Delta indicators with semantic colors */
    .metric-delta-positive {{
        color: {COLORS["GREEN"]};
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 10px;
    }}

    .metric-delta-negative {{
        color: {COLORS["RED"]};
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 10px;
    }}

    .metric-delta-neutral {{
        color: {COLORS["NEUTRAL"]};
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 10px;
    }}

    /* Context information */
    .metric-context {{
        font-size: 0.85rem;
        color: #444;
        margin-top: 6px;
        line-height: 1.4;
    }}

    /* Highlight important context */
    .metric-context-highlight {{
        font-weight: 600;
        color: #333;
    }}

    /* Status indicators */
    .status-indicator {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
    }}

    .status-positive {{
        background-color: rgba(44, 160, 44, 0.15);
        color: {COLORS["GREEN"]};
    }}

    .status-negative {{
        background-color: rgba(214, 39, 40, 0.15);
        color: {COLORS["RED"]};
    }}

    .status-neutral {{
        background-color: rgba(144, 144, 144, 0.15);
        color: {COLORS["NEUTRAL"]};
    }}

    .status-warning {{
        background-color: rgba(225, 129, 60, 0.15);
        color: {COLORS["ORANGE"]};
    }}

    /* Card with positive border */
    .metric-card-positive {{
        border-left: 4px solid {COLORS["GREEN"]};
    }}

    /* Card with negative border */
    .metric-card-negative {{
        border-left: 4px solid {COLORS["RED"]};
    }}

    /* Card with neutral border */
    .metric-card-neutral {{
        border-left: 4px solid {COLORS["NEUTRAL"]};
    }}

    /* Card with warning border */
    .metric-card-warning {{
        border-left: 4px solid {COLORS["ORANGE"]};
    }}
    </style>
    """, unsafe_allow_html=True)

    # Create a 2x2 grid for the main KPIs
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # KPI 1: Price Change with trend indicator
    price_change_pct_overview = (price.iloc[-1] / price.iloc[-2] - 1) if len(price) >= 2 else None
    price_change_abs = price.iloc[-1] - price.iloc[-2] if len(price) >= 2 else None

    # Determine the trend over the last 5 days
    trend_status = "neutral"
    trend_text = "Neutral trend"
    if len(price) >= 5:
        five_day_change = price.iloc[-1] / price.iloc[-5] - 1
        if five_day_change > 0.03:
            trend_status = "positive"
            trend_text = "Strong upward trend"
        elif five_day_change > 0.01:
            trend_status = "positive"
            trend_text = "Upward trend"
        elif five_day_change < -0.03:
            trend_status = "negative"
            trend_text = "Strong downward trend"
        elif five_day_change < -0.01:
            trend_status = "negative"
            trend_text = "Downward trend"
        else:
            trend_status = "neutral"
            trend_text = "Sideways trend"

    with col1:
        # Determine card style based on price change
        card_class = "metric-card"
        if price_change_pct_overview:
            if price_change_pct_overview > 0:
                card_class = "metric-card metric-card-positive"
                delta_class = "metric-delta-positive"
                delta_symbol = "↑"
            else:
                card_class = "metric-card metric-card-negative"
                delta_class = "metric-delta-negative"
                delta_symbol = "↓"
        else:
            delta_class = "metric-delta-neutral"
            delta_symbol = "–"

        delta_text = f"{delta_symbol} {format_pct(abs(price_change_pct_overview))}" if price_change_pct_overview is not None else ""

        # Create status indicator for the trend
        status_class = f"status-{trend_status}"

        st.markdown(f"""
        <div class="{card_class}">
            <div class="metric-title">DAY CHANGE <span class="{status_class} status-indicator">{trend_text}</span></div>
            <div class="metric-value">{format_pct(price_change_pct_overview) if price_change_pct_overview is not None else "–"}</div>
            <div class="{delta_class}">{delta_text}</div>
            <div class="metric-context">Absolute change: ${f"{price_change_abs:.2f}" if price_change_abs is not None else "0.00"}</div>
            <div class="metric-context">Based on 5-day price movement</div>
        </div>
        """, unsafe_allow_html=True)

    # KPI 2: Market Cap with category indicator
    market_cap = info.get('marketCap')
    shares_outstanding = info.get('sharesOutstanding')

    # Market cap categories with context
    cap_category = ""
    cap_status = "neutral"
    cap_context = ""

    if market_cap:
        if market_cap > 200e9:
            cap_category = "Mega Cap"
            cap_status = "positive"
            cap_context = "Top 1% of companies by size"
        elif market_cap > 10e9:
            cap_category = "Large Cap"
            cap_status = "positive"
            cap_context = "Top 10% of companies by size"
        elif market_cap > 2e9:
            cap_category = "Mid Cap"
            cap_status = "neutral"
            cap_context = "Medium-sized company"
        elif market_cap > 300e6:
            cap_category = "Small Cap"
            cap_status = "neutral"
            cap_context = "Smaller company"
        else:
            cap_category = "Micro Cap"
            cap_status = "warning"
            cap_context = "Very small company"

    # Calculate shares outstanding context if available
    shares_context = ""
    if shares_outstanding and market_cap:
        price_per_share = market_cap / shares_outstanding
        shares_context = f"Implied share price: ${price_per_share:.2f}"

    with col2:
        # Determine card style based on market cap category
        card_class = "metric-card"
        if cap_status == "positive":
            card_class = "metric-card metric-card-positive"
        elif cap_status == "warning":
            card_class = "metric-card metric-card-warning"

        # Status indicator for market cap category
        status_class = f"status-{cap_status}"

        st.markdown(f"""
        <div class="{card_class}">
            <div class="metric-title">MARKET CAP <span class="{status_class} status-indicator">{cap_category}</span></div>
            <div class="metric-value">{format_with_suffix(market_cap) if market_cap else "–"}</div>
            <div class="metric-context-highlight">{cap_context if market_cap else ""}</div>
            <div class="metric-context">Exact value: {format_currency(market_cap, 0) if market_cap else "–"}</div>
            <div class="metric-context">{shares_context}</div>
        </div>
        """, unsafe_allow_html=True)

    # KPI 3: P/E Ratio with valuation indicator
    pe_ratio = info.get('trailingPE')
    forward_pe = info.get('forwardPE')
    sector = info.get('sector')

    # Define sector average P/Es (simplified for illustration)
    sector_pe_map = {
        "Technology": 30,
        "Healthcare": 25,
        "Consumer Cyclical": 22,
        "Financial Services": 15,
        "Communication Services": 20,
        "Industrials": 18,
        "Consumer Defensive": 20,
        "Energy": 12,
        "Basic Materials": 14,
        "Utilities": 16,
        "Real Estate": 18
    }

    # Market average P/E (simplified)
    market_avg_pe = 20

    with col3:
        pe_context = ""
        pe_status = "neutral"
        sector_avg = sector_pe_map.get(sector, market_avg_pe) if sector else market_avg_pe

        # Determine if P/E is high, low, or average compared to sector
        if pe_ratio:
            pe_diff_pct = (pe_ratio / sector_avg - 1) * 100

            if pe_diff_pct < -30:
                pe_context = f"Significantly undervalued vs. {sector or 'market'}"
                pe_status = "positive"
                valuation = "Undervalued"
            elif pe_diff_pct < -15:
                pe_context = f"Potentially undervalued vs. {sector or 'market'}"
                pe_status = "positive"
                valuation = "Undervalued"
            elif pe_diff_pct > 50:
                pe_context = f"Significantly overvalued vs. {sector or 'market'}"
                pe_status = "negative"
                valuation = "Overvalued"
            elif pe_diff_pct > 20:
                pe_context = f"Potentially overvalued vs. {sector or 'market'}"
                pe_status = "warning"
                valuation = "Potentially Overvalued"
            else:
                pe_context = f"Fairly valued vs. {sector or 'market'}"
                pe_status = "neutral"
                valuation = "Fair Value"

        # Calculate P/E to growth ratio if forward P/E is available
        peg_context = ""
        if forward_pe and info.get('pegRatio'):
            peg = info.get('pegRatio')
            if peg < 1:
                peg_context = "PEG < 1: Potentially undervalued"
            elif peg < 1.5:
                peg_context = "PEG 1-1.5: Fairly valued"
            else:
                peg_context = "PEG > 1.5: Potentially overvalued"

        # Determine card style based on P/E status
        card_class = "metric-card"
        if pe_status == "positive":
            card_class = "metric-card metric-card-positive"
        elif pe_status == "negative":
            card_class = "metric-card metric-card-negative"
        elif pe_status == "warning":
            card_class = "metric-card metric-card-warning"

        # Status indicator
        status_class = f"status-{pe_status}"

        st.markdown(f"""
        <div class="{card_class}">
            <div class="metric-title">P/E RATIO (TTM) <span class="{status_class} status-indicator">{valuation if pe_ratio else ''}</span></div>
            <div class="metric-value">{f"{pe_ratio:.2f}" if pe_ratio else "–"}</div>
            <div class="metric-context-highlight">{pe_context}</div>
            <div class="metric-context">Forward P/E: {f"{forward_pe:.2f}" if forward_pe else "–"}</div>
            <div class="metric-context">{peg_context}</div>
            <div class="metric-context">Sector avg: {f"{sector_avg:.2f}" if sector else f"{market_avg_pe:.2f} (market)"}</div>
        </div>
        """, unsafe_allow_html=True)

    # KPI 4: Dividend Yield with sustainability indicator
    dividend_yield = info.get('dividendYield')
    payout_ratio = info.get('payoutRatio')
    dividend_rate = info.get('dividendRate')

    with col4:
        # Determine dividend status and context
        dividend_status = "neutral"
        sustainability = ""

        if dividend_yield is None or dividend_yield == 0:
            dividend_context = "No dividend payments"
            dividend_status = "neutral"
            yield_status = ""
        else:
            # Evaluate dividend yield
            if dividend_yield > 0.06:
                dividend_context = "Very high yield stock"
                dividend_status = "warning"  # High yields can be unsustainable
                yield_status = "Very High Yield"
            elif dividend_yield > 0.04:
                dividend_context = "High yield stock"
                dividend_status = "positive"
                yield_status = "High Yield"
            elif dividend_yield > 0.02:
                dividend_context = "Above average yield"
                dividend_status = "positive"
                yield_status = "Above Avg Yield"
            else:
                dividend_context = "Modest dividend yield"
                dividend_status = "neutral"
                yield_status = "Modest Yield"

            # Evaluate dividend sustainability based on payout ratio
            if payout_ratio:
                if payout_ratio > 0.9:
                    sustainability = "Potentially unsustainable payout"
                    dividend_status = "negative"
                elif payout_ratio > 0.7:
                    sustainability = "High payout ratio"
                    dividend_status = "warning"
                elif payout_ratio > 0.5:
                    sustainability = "Moderate payout ratio"
                    dividend_status = "neutral"
                else:
                    sustainability = "Sustainable payout ratio"
                    dividend_status = "positive"

        # Determine card style based on dividend status
        card_class = "metric-card"
        if dividend_status == "positive":
            card_class = "metric-card metric-card-positive"
        elif dividend_status == "negative":
            card_class = "metric-card metric-card-negative"
        elif dividend_status == "warning":
            card_class = "metric-card metric-card-warning"

        # Status indicator
        status_class = f"status-{dividend_status}"

        st.markdown(f"""
        <div class="{card_class}">
            <div class="metric-title">DIVIDEND YIELD <span class="{status_class} status-indicator">{yield_status}</span></div>
            <div class="metric-value">{format_pct(dividend_yield) if dividend_yield else "–"}</div>
            <div class="metric-context-highlight">{dividend_context}</div>
            <div class="metric-context">{sustainability}</div>
            <div class="metric-context">Payout ratio: {format_pct(payout_ratio) if payout_ratio else "–"}</div>
            <div class="metric-context">Annual dividend: ${f"{dividend_rate:.2f}" if dividend_rate else "–"} per share</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Additional KPIs ---
    st.subheader("Additional Metrics")

    # Create a 2x2 grid for additional metrics
    acol1, acol2 = st.columns(2)
    acol3, acol4 = st.columns(2)

    # 52 Week High/Low with momentum indicator
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

            # Determine momentum status based on position in range
            momentum_status = "neutral"
            momentum_text = "Neutral momentum"

            if range_position > 80:
                momentum_status = "positive"
                momentum_text = "Strong bullish momentum"
            elif range_position > 60:
                momentum_status = "positive"
                momentum_text = "Bullish momentum"
            elif range_position < 20:
                momentum_status = "negative"
                momentum_text = "Strong bearish momentum"
            elif range_position < 40:
                momentum_status = "negative"
                momentum_text = "Bearish momentum"

            # Determine card style based on momentum
            card_class = "metric-card"
            if momentum_status == "positive":
                card_class = "metric-card metric-card-positive"
            elif momentum_status == "negative":
                card_class = "metric-card metric-card-negative"

            # Status indicator
            status_class = f"status-{momentum_status}"

            # Create a historical context for the current position
            historical_context = ""
            if range_position > 90:
                historical_context = "Price is near the 52-week high, indicating strong recent performance"
            elif range_position > 70:
                historical_context = "Price is in the upper range, showing positive momentum"
            elif range_position < 10:
                historical_context = "Price is near the 52-week low, indicating weak recent performance"
            elif range_position < 30:
                historical_context = "Price is in the lower range, showing negative momentum"
            else:
                historical_context = "Price is in the middle of its 52-week range"

            st.markdown(f"""
            <div class="{card_class}">
                <div class="metric-title">52-WEEK RANGE <span class="{status_class} status-indicator">{momentum_text}</span></div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px; color: black;">
                    <div style="text-align: left; font-weight: 600; color: black;">Low: ${week_low:.2f}</div>
                    <div style="text-align: right; font-weight: 600; color: black;">High: ${week_high:.2f}</div>
                </div>
                <div style="height: 6px; background-color: #eee; border-radius: 3px; margin-bottom: 10px;">
                    <div style="width: {range_position}%; height: 100%; background-color: {COLORS["BLUE"]}; border-radius: 3px;"></div>
                </div>
                <div class="metric-context-highlight">{historical_context}</div>
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

    # Volume metrics with trend indicator
    avg_volume = info.get('averageVolume')
    current_volume = volume.iloc[-1] if not volume.empty else None

    # Calculate volume trend (average of last 3 days vs average of previous 7 days)
    volume_trend = None
    if len(volume) >= 10:
        recent_volumes = volume.iloc[-10:].tolist()
        recent_avg = sum(recent_volumes[-3:]) / 3
        prev_avg = sum(recent_volumes[:-3]) / 7
        volume_trend = (recent_avg / prev_avg - 1) * 100 if prev_avg > 0 else 0

    with acol2:
        # Determine volume status and context
        volume_status = "neutral"
        volume_context = ""

        if avg_volume and current_volume:
            volume_ratio = current_volume / avg_volume
            if volume_ratio > 3:
                volume_context = "Extremely high volume today"
                volume_status = "warning"  # Unusual activity might indicate volatility
                volume_indicator = "Extremely High"
            elif volume_ratio > 2:
                volume_context = "Very high volume today"
                volume_status = "warning"
                volume_indicator = "Very High"
            elif volume_ratio > 1.5:
                volume_context = "Above average volume today"
                volume_status = "positive"
                volume_indicator = "Above Average"
            elif volume_ratio < 0.5:
                volume_context = "Very low volume today"
                volume_status = "negative"
                volume_indicator = "Very Low"
            elif volume_ratio < 0.7:
                volume_context = "Below average volume today"
                volume_status = "neutral"
                volume_indicator = "Below Average"
            else:
                volume_context = "Normal trading volume"
                volume_status = "neutral"
                volume_indicator = "Normal"
        else:
            volume_indicator = ""

        # Determine card style based on volume status
        card_class = "metric-card"
        if volume_status == "positive":
            card_class = "metric-card metric-card-positive"
        elif volume_status == "negative":
            card_class = "metric-card metric-card-negative"
        elif volume_status == "warning":
            card_class = "metric-card metric-card-warning"

        # Status indicator
        status_class = f"status-{volume_status}"

        # Volume trend text
        trend_text = ""
        if volume_trend is not None:
            if volume_trend > 20:
                trend_text = "Strongly increasing volume trend"
            elif volume_trend > 10:
                trend_text = "Increasing volume trend"
            elif volume_trend < -20:
                trend_text = "Strongly decreasing volume trend"
            elif volume_trend < -10:
                trend_text = "Decreasing volume trend"
            else:
                trend_text = "Stable volume trend"

        st.markdown(f"""
        <div class="{card_class}">
            <div class="metric-title">VOLUME METRICS <span class="{status_class} status-indicator">{volume_indicator}</span></div>
            <div class="metric-value">{format_num(avg_volume) if avg_volume else "–"}</div>
            <div class="metric-context-highlight">{volume_context}</div>
            <div class="metric-context">Today: {format_num(current_volume) if current_volume else "–"}</div>
            <div class="metric-context">Ratio to avg: {f"{volume_ratio:.2f}x" if avg_volume and current_volume else "–"}</div>
            <div class="metric-context">{trend_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # Beta with risk indicator
    beta = info.get('beta')

    with acol3:
        # Determine beta status and context
        beta_status = "neutral"
        beta_context = ""
        risk_level = ""

        if beta is not None:
            if beta > 2:
                beta_context = "Extremely high volatility compared to market"
                beta_status = "warning"
                risk_level = "Very High Risk"
            elif beta > 1.5:
                beta_context = "High volatility compared to market"
                beta_status = "warning"
                risk_level = "High Risk"
            elif beta > 1.2:
                beta_context = "Above average volatility"
                beta_status = "neutral"
                risk_level = "Above Average Risk"
            elif beta > 0.8:
                beta_context = "Similar volatility to market"
                beta_status = "neutral"
                risk_level = "Average Risk"
            elif beta > 0.5:
                beta_context = "Below average volatility"
                beta_status = "positive"
                risk_level = "Below Average Risk"
            elif beta > 0:
                beta_context = "Low volatility compared to market"
                beta_status = "positive"
                risk_level = "Low Risk"
            elif beta > -0.5:
                beta_context = "Slight negative correlation with market"
                beta_status = "neutral"
                risk_level = "Hedging Potential"
            else:
                beta_context = "Strong negative correlation with market"
                beta_status = "neutral"
                risk_level = "Strong Hedging Potential"

        # Determine card style based on beta status
        card_class = "metric-card"
        if beta_status == "positive":
            card_class = "metric-card metric-card-positive"
        elif beta_status == "warning":
            card_class = "metric-card metric-card-warning"

        # Status indicator
        status_class = f"status-{beta_status}"

        # Investment style context based on beta
        investment_context = ""
        if beta is not None:
            if beta > 1.5:
                investment_context = "Suitable for aggressive growth investors"
            elif beta > 1:
                investment_context = "Suitable for growth-oriented investors"
            elif beta > 0.8:
                investment_context = "Suitable for balanced portfolios"
            elif beta > 0:
                investment_context = "Suitable for conservative investors"
            else:
                investment_context = "Potential hedge against market downturns"

        st.markdown(f"""
        <div class="{card_class}">
            <div class="metric-title">BETA <span class="{status_class} status-indicator">{risk_level}</span></div>
            <div class="metric-value">{f"{beta:.2f}" if beta is not None else "–"}</div>
            <div class="metric-context-highlight">{beta_context}</div>
            <div class="metric-context">{investment_context}</div>
            <div class="metric-context">Market beta = 1.0 (reference point)</div>
        </div>
        """, unsafe_allow_html=True)

    # Analyst recommendations with consensus indicator
    target_price = info.get('targetMeanPrice')
    target_high = info.get('targetHighPrice')
    target_low = info.get('targetLowPrice')
    recommendation_key = info.get('recommendationKey', '').lower()

    with acol4:
        # Determine analyst consensus and context
        analyst_status = "neutral"
        consensus_text = "Neutral"

        if target_price and current_price:
            upside = (target_price / current_price - 1) * 100

            if upside > 30:
                analyst_context = "Extremely bullish outlook"
                analyst_status = "positive"
                consensus_text = "Strong Buy"
            elif upside > 20:
                analyst_context = "Very bullish outlook"
                analyst_status = "positive"
                consensus_text = "Buy"
            elif upside > 10:
                analyst_context = "Bullish outlook"
                analyst_status = "positive"
                consensus_text = "Outperform"
            elif upside > 0:
                analyst_context = "Slightly bullish outlook"
                analyst_status = "neutral"
                consensus_text = "Mild Outperform"
            elif upside > -10:
                analyst_context = "Neutral outlook"
                analyst_status = "neutral"
                consensus_text = "Hold"
            elif upside > -20:
                analyst_context = "Slightly bearish outlook"
                analyst_status = "warning"
                consensus_text = "Mild Underperform"
            else:
                analyst_context = "Bearish outlook"
                analyst_status = "negative"
                consensus_text = "Underperform"
        else:
            analyst_context = ""

        # Override consensus text with actual recommendation if available
        if recommendation_key:
            if recommendation_key == 'buy':
                consensus_text = "Buy"
                analyst_status = "positive"
            elif recommendation_key == 'overweight':
                consensus_text = "Overweight"
                analyst_status = "positive"
            elif recommendation_key == 'hold':
                consensus_text = "Hold"
                analyst_status = "neutral"
            elif recommendation_key == 'underweight':
                consensus_text = "Underweight"
                analyst_status = "warning"
            elif recommendation_key == 'sell':
                consensus_text = "Sell"
                analyst_status = "negative"

        # Determine card style based on analyst status
        card_class = "metric-card"
        if analyst_status == "positive":
            card_class = "metric-card metric-card-positive"
        elif analyst_status == "negative":
            card_class = "metric-card metric-card-negative"
        elif analyst_status == "warning":
            card_class = "metric-card metric-card-warning"

        # Status indicator
        status_class = f"status-{analyst_status}"

        st.markdown(f"""
        <div class="{card_class}">
            <div class="metric-title">ANALYST TARGETS <span class="{status_class} status-indicator">{consensus_text}</span></div>
            <div class="metric-value">${f"{target_price:.2f}" if target_price else "–"}</div>
            <div class="metric-context-highlight">{analyst_context}</div>
            <div class="metric-context">Range: ${f"{target_low:.2f}" if target_low else "–"} - ${f"{target_high:.2f}" if target_high else "–"}</div>
            <div class="metric-context">Potential upside: {f"{upside:.1f}%" if target_price and current_price else "–"}</div>
            <div class="metric-context">Based on analyst consensus</div>
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

    # Volatility with risk indicator
    volatility = metrics.get('vol')

    # Calculate recent (30-day) volatility if we have enough data
    recent_volatility = None
    market_volatility = 0.15  # Simplified market volatility (VIX/100) for comparison

    if 'rets' in metrics and not metrics['rets'].empty and len(metrics['rets']) >= 30:
        recent_volatility = metrics['rets'].iloc[-30:].std() * np.sqrt(252)

    with rcol1:
        # Determine volatility status and context
        vol_status = "neutral"
        if pd.notna(volatility):
            if volatility > 0.5:
                vol_context = "Extremely high volatility"
                vol_status = "negative"
                risk_level = "Very High Risk"
            elif volatility > 0.4:
                vol_context = "Very high volatility"
                vol_status = "negative"
                risk_level = "High Risk"
            elif volatility > 0.3:
                vol_context = "High volatility"
                vol_status = "warning"
                risk_level = "Above Average Risk"
            elif volatility > 0.2:
                vol_context = "Moderate volatility"
                vol_status = "neutral"
                risk_level = "Average Risk"
            elif volatility > 0.15:
                vol_context = "Below average volatility"
                vol_status = "positive"
                risk_level = "Below Average Risk"
            else:
                vol_context = "Low volatility"
                vol_status = "positive"
                risk_level = "Low Risk"
        else:
            vol_context = ""
            risk_level = ""

        # Calculate relative volatility (stock vol / market vol)
        rel_vol_text = ""
        if pd.notna(volatility) and market_volatility:
            rel_vol = volatility / market_volatility
            rel_vol_text = f"{rel_vol:.1f}x market volatility"

        # Determine card style based on volatility status
        card_class = "metric-card"
        if vol_status == "positive":
            card_class = "metric-card metric-card-positive"
        elif vol_status == "negative":
            card_class = "metric-card metric-card-negative"
        elif vol_status == "warning":
            card_class = "metric-card metric-card-warning"

        # Status indicator
        status_class = f"status-{vol_status}"

        st.markdown(f"""
        <div class="{card_class}">
            <div class="metric-title">VOLATILITY (ANNUALIZED) <span class="{status_class} status-indicator">{risk_level}</span></div>
            <div class="metric-value">{format_pct(volatility) if pd.notna(volatility) else "–"}</div>
            <div class="metric-context-highlight">{vol_context}</div>
            <div class="metric-context">{rel_vol_text}</div>
            <div class="metric-context">Recent (30-day) volatility: {format_pct(recent_volatility) if pd.notna(recent_volatility) else "–"}</div>
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
