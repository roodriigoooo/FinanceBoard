"""
Technical analysis page for the Finance Board application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from modules.visualizations.charts import create_candlestick_chart
from config.settings import COLORS


def render_technical_page(ticker, asset_name, df, price, volume, metrics, bench_price_col=None, bdf=None, benchmark=None):
    """
    Render the technical analysis page.

    Args:
        ticker: Ticker symbol
        asset_name: Asset name
        df: DataFrame with OHLC data
        price: Series with price data
        volume: Series with volume data
        metrics: Dictionary with metrics
        bench_price_col: Column name for benchmark price
        bdf: DataFrame with benchmark data
        benchmark: Benchmark ticker symbol
    """
    st.header(f"{asset_name} ({ticker}) - Technical Analysis")

    # Add a general disclaimer for all technical indicators
    st.info("""
    **Disclaimer:** All technical indicators and signals shown on this page are for informational purposes only and should not be
    considered as investment advice. Technical analysis has inherent limitations and past performance is not indicative of future results.
    Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
    """)

    # --- Price chart (Candlestick) ---
    if not df.empty and all(
            col in df.columns for col in ['Open', 'High', 'Low', 'Close']) and not price.empty and not volume.empty:

        fig_tech = create_candlestick_chart(df, price, volume, metrics, bench_price_col, bdf, benchmark, ticker, asset_name)
        st.plotly_chart(fig_tech, use_container_width=True)
    else:
        st.warning(
            f"Could not display full technical chart. Required data (OHLC, Price, Volume) missing or insufficient for {ticker}.")

    # --- Technical Indicators ---
    st.subheader("Technical Indicators")

    # Create tabs for different technical indicators
    tech_tabs = st.tabs(["Moving Averages", "Oscillators", "Volatility", "Volume Analysis"])

    with tech_tabs[0]:  # Moving Averages
        st.markdown("### Moving Averages")

        # Allow user to select moving average periods
        ma_col1, ma_col2 = st.columns(2)
        with ma_col1:
            short_ma = st.slider("Short MA Period", min_value=5, max_value=50, value=20, step=1)
        with ma_col2:
            long_ma = st.slider("Long MA Period", min_value=20, max_value=200, value=50, step=5)

        # Calculate moving averages
        if not price.empty:
            short_ma_values = price.rolling(window=short_ma).mean()
            long_ma_values = price.rolling(window=long_ma).mean()

            # Create figure
            fig_ma = go.Figure()

            # Add price
            fig_ma.add_trace(go.Scatter(
                x=price.index,
                y=price,
                name=f"{ticker} Price",
                line=dict(color=COLORS["NEUTRAL"])
            ))

            # Add moving averages
            fig_ma.add_trace(go.Scatter(
                x=short_ma_values.index,
                y=short_ma_values,
                name=f"{short_ma}-day MA",
                line=dict(color=COLORS["BLUE"])
            ))

            fig_ma.add_trace(go.Scatter(
                x=long_ma_values.index,
                y=long_ma_values,
                name=f"{long_ma}-day MA",
                line=dict(color=COLORS["ORANGE"])
            ))

            # Update layout
            fig_ma.update_layout(
                title=f"Moving Averages ({short_ma}-day and {long_ma}-day)",
                xaxis_title="Date",
                yaxis_title="Price",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=50, b=0)
            )

            st.plotly_chart(fig_ma, use_container_width=True)

            # Moving Average Crossover Analysis
            st.subheader("Moving Average Crossover Analysis")

            # Calculate crossover signals
            signal = pd.Series(0, index=price.index)
            signal[short_ma_values > long_ma_values] = 1  # Buy signal
            signal[short_ma_values < long_ma_values] = -1  # Sell signal

            # Detect crossovers (signal changes)
            crossovers = signal.diff().fillna(0)
            buy_signals = price[crossovers == 1]  # Buy when short MA crosses above long MA
            sell_signals = price[crossovers == -1]  # Sell when short MA crosses below long MA

            # Current signal
            current_signal = signal.iloc[-1]
            signal_text = "NEUTRAL"

            if current_signal == 1:
                signal_text = "BULLISH"
            elif current_signal == -1:
                signal_text = "BEARISH"

            # Use a consistent blue color for all signals
            signal_hex_color = COLORS["BLUE"]

            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; background-color:{signal_hex_color}; color:white; text-align:center; margin-bottom:20px;">
                <strong>Current MA Crossover Signal: {signal_text}</strong>
                <p style="font-size:0.8rem; margin-top:5px; margin-bottom:0;">
                This technical indicator is based solely on the relationship between the {short_ma}-day and {long_ma}-day moving averages.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Add a help box with disclaimer and explanation
            with st.expander("ℹ️ About Moving Average Crossover Signals"):
                st.markdown("""
                **Disclaimer:** The signals shown are purely technical indicators and should not be considered as investment advice.
                Technical analysis has limitations and should be used alongside other forms of analysis.

                **What this means:**
                - **Bullish Signal:** The short-term moving average is above the long-term moving average, which traditionally suggests upward momentum.
                - **Bearish Signal:** The short-term moving average is below the long-term moving average, which traditionally suggests downward momentum.
                - **Neutral Signal:** The moving averages are very close to each other, suggesting no clear trend direction.

                **Limitations:**
                - Moving average crossovers are lagging indicators and may generate false signals in choppy markets.
                - They work best in trending markets and may not be reliable in sideways or highly volatile conditions.
                - Past performance is not indicative of future results.

                Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
                """)

            # Show recent crossovers
            if not buy_signals.empty or not sell_signals.empty:
                st.markdown("#### Recent Crossovers")

                # Combine and sort signals
                # Convert DatetimeIndex to list before creating DataFrame
                buy_dates = buy_signals.index.tolist() if not buy_signals.empty else []
                sell_dates = sell_signals.index.tolist() if not sell_signals.empty else []
                buy_prices = buy_signals.tolist() if not buy_signals.empty else []
                sell_prices = sell_signals.tolist() if not sell_signals.empty else []

                all_signals = pd.DataFrame({
                    'Date': buy_dates + sell_dates,
                    'Price': buy_prices + sell_prices,
                    'Signal': ['Buy'] * len(buy_signals) + ['Sell'] * len(sell_signals)
                })
                all_signals = all_signals.sort_values('Date', ascending=False).head(5)

                # Display in a table
                st.dataframe(all_signals.set_index('Date'))
            else:
                st.info("No crossover signals detected in the selected time period.")
        else:
            st.warning(f"Insufficient price data for {ticker} to calculate moving averages.")

    with tech_tabs[1]:  # Oscillators
        st.markdown("### Oscillators")

        # RSI
        if "rsi_series" in metrics and not metrics["rsi_series"].empty:
            rsi = metrics["rsi_series"]

            # Create RSI figure
            fig_rsi = go.Figure()

            fig_rsi.add_trace(go.Scatter(
                x=rsi.index,
                y=rsi,
                name="RSI",
                line=dict(color=COLORS["PURPLE"])
            ))

            # Add overbought/oversold lines
            fig_rsi.add_hline(y=70, line_dash="dash", line_color=COLORS["RED"], annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color=COLORS["GREEN"], annotation_text="Oversold")

            # Update layout
            fig_rsi.update_layout(
                title="Relative Strength Index (RSI)",
                xaxis_title="Date",
                yaxis_title="RSI",
                yaxis=dict(range=[0, 100]),
                margin=dict(l=0, r=0, t=50, b=0)
            )

            st.plotly_chart(fig_rsi, use_container_width=True)

            # RSI interpretation
            current_rsi = rsi.iloc[-1] if not rsi.empty else None

            if current_rsi is not None:
                rsi_status = "Neutral"
                rsi_description = "The asset is neither overbought nor oversold."

                if current_rsi > 70:
                    rsi_status = "Overbought"
                    rsi_description = "Traditionally suggests the asset may be overvalued."
                elif current_rsi < 30:
                    rsi_status = "Oversold"
                    rsi_description = "Traditionally suggests the asset may be undervalued."

                # Use blue for all RSI indicators
                st.markdown(f"""
                <div style="padding:15px; border-radius:8px; background-color:{COLORS["BLUE"]}; color:white;">
                    <strong>RSI: {current_rsi:.2f} - {rsi_status}</strong> - {rsi_description}
                    <p style="font-size:0.8rem; margin-top:5px; margin-bottom:0;">
                    This is a technical indicator and not investment advice.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Add a help box with disclaimer and explanation
                with st.expander("ℹ️ About RSI (Relative Strength Index)"):
                    st.markdown("""
                    **Disclaimer:** The RSI indicator shown is purely technical and should not be considered as investment advice.
                    Technical analysis has limitations and should be used alongside other forms of analysis.

                    **What this means:**
                    - **Overbought (RSI > 70):** Traditionally suggests that the asset may be overvalued and could experience a price correction.
                    - **Oversold (RSI < 30):** Traditionally suggests that the asset may be undervalued and could experience a price increase.
                    - **Neutral (30 < RSI < 70):** The asset is trading in a normal range according to this indicator.

                    **Limitations:**
                    - RSI can remain in overbought or oversold territory for extended periods during strong trends.
                    - False signals are common, especially in trending markets.
                    - RSI works best in ranging markets and may be less reliable during strong trends.
                    - Past performance is not indicative of future results.

                    Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
                    """)
        else:
            st.warning(f"Insufficient data for {ticker} to calculate RSI.")

        # MACD
        if all(k in metrics for k in ["macd_line", "signal_line", "macd_histogram"]):
            macd_line = metrics["macd_line"]
            signal_line = metrics["signal_line"]
            macd_histogram = metrics["macd_histogram"]

            if not macd_line.empty and not signal_line.empty and not macd_histogram.empty:
                # Create MACD figure
                fig_macd = go.Figure()

                fig_macd.add_trace(go.Scatter(
                    x=macd_line.index,
                    y=macd_line,
                    name="MACD Line",
                    line=dict(color=COLORS["BLUE"])
                ))

                fig_macd.add_trace(go.Scatter(
                    x=signal_line.index,
                    y=signal_line,
                    name="Signal Line",
                    line=dict(color=COLORS["ORANGE"])
                ))

                # Add histogram
                colors = [COLORS["GREEN"] if val >= 0 else COLORS["RED"] for val in macd_histogram]
                fig_macd.add_trace(go.Bar(
                    x=macd_histogram.index,
                    y=macd_histogram,
                    name="MACD Histogram",
                    marker_color=colors
                ))

                # Update layout
                fig_macd.update_layout(
                    title="Moving Average Convergence Divergence (MACD)",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=0, r=0, t=50, b=0)
                )

                st.plotly_chart(fig_macd, use_container_width=True)

                # MACD interpretation
                current_macd = macd_line.iloc[-1] if not macd_line.empty else None
                current_signal = signal_line.iloc[-1] if not signal_line.empty else None

                if current_macd is not None and current_signal is not None:
                    macd_status = "Bullish" if current_macd > current_signal else "Bearish"
                    macd_description = "MACD line is above the signal line." if current_macd > current_signal else "MACD line is below the signal line."

                    # Use blue for all MACD indicators
                    st.markdown(f"""
                    <div style="padding:15px; border-radius:8px; background-color:{COLORS["BLUE"]}; color:white;">
                        <strong>MACD: {macd_status}</strong> - {macd_description}
                        <p style="font-size:0.8rem; margin-top:5px; margin-bottom:0;">
                        This is a technical indicator and not investment advice.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Add a help box with disclaimer and explanation
                    with st.expander("ℹ️ About MACD (Moving Average Convergence Divergence)"):
                        st.markdown("""
                        **Disclaimer:** The MACD indicator shown is purely technical and should not be considered as investment advice.
                        Technical analysis has limitations and should be used alongside other forms of analysis.

                        **What this means:**
                        - **Bullish Signal:** The MACD line is above the signal line, which traditionally suggests upward momentum.
                        - **Bearish Signal:** The MACD line is below the signal line, which traditionally suggests downward momentum.

                        **Limitations:**
                        - MACD is a lagging indicator and may generate delayed signals.
                        - False signals can occur, especially in choppy or sideways markets.
                        - MACD works best in trending markets and may be less reliable during range-bound conditions.
                        - Past performance is not indicative of future results.

                        Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
                        """)
            else:
                st.warning(f"Insufficient data for {ticker} to calculate MACD.")
        else:
            st.warning(f"Insufficient data for {ticker} to calculate MACD.")

    with tech_tabs[2]:  # Volatility
        st.markdown("### Volatility Analysis")

        if not price.empty:
            # Calculate rolling volatility
            returns = price.pct_change().dropna()
            rolling_vol_20 = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
            rolling_vol_60 = returns.rolling(window=60).std() * np.sqrt(252)  # Annualized

            # Create volatility figure
            fig_vol = go.Figure()

            fig_vol.add_trace(go.Scatter(
                x=rolling_vol_20.index,
                y=rolling_vol_20,
                name="20-day Volatility",
                line=dict(color=COLORS["BLUE"])
            ))

            fig_vol.add_trace(go.Scatter(
                x=rolling_vol_60.index,
                y=rolling_vol_60,
                name="60-day Volatility",
                line=dict(color=COLORS["ORANGE"])
            ))

            # Update layout
            fig_vol.update_layout(
                title="Rolling Volatility (Annualized)",
                xaxis_title="Date",
                yaxis_title="Volatility",
                yaxis_tickformat=".0%",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=50, b=0)
            )

            st.plotly_chart(fig_vol, use_container_width=True)

            # Bollinger Bands
            st.markdown("#### Bollinger Bands")

            # Allow user to select Bollinger Band parameters
            bb_col1, bb_col2 = st.columns(2)
            with bb_col1:
                bb_period = st.slider("Bollinger Band Period", min_value=5, max_value=50, value=20, step=1)
            with bb_col2:
                bb_std = st.slider("Standard Deviation Multiplier", min_value=1.0, max_value=3.0, value=2.0, step=0.1)

            # Calculate Bollinger Bands
            bb_ma = price.rolling(window=bb_period).mean()
            bb_std_dev = price.rolling(window=bb_period).std()
            upper_bb = bb_ma + (bb_std_dev * bb_std)
            lower_bb = bb_ma - (bb_std_dev * bb_std)

            # Create Bollinger Bands figure
            fig_bb = go.Figure()

            fig_bb.add_trace(go.Scatter(
                x=price.index,
                y=price,
                name=f"{ticker} Price",
                line=dict(color=COLORS["NEUTRAL"])
            ))

            fig_bb.add_trace(go.Scatter(
                x=bb_ma.index,
                y=bb_ma,
                name="Middle Band (SMA)",
                line=dict(color=COLORS["BLUE"], dash="dash")
            ))

            fig_bb.add_trace(go.Scatter(
                x=upper_bb.index,
                y=upper_bb,
                name="Upper Band",
                line=dict(color=COLORS["RED"], width=0.7)
            ))

            fig_bb.add_trace(go.Scatter(
                x=lower_bb.index,
                y=lower_bb,
                name="Lower Band",
                line=dict(color=COLORS["GREEN"], width=0.7),
                fill='tonexty',
                fillcolor='rgba(0, 100, 80, 0.1)'
            ))

            # Update layout
            fig_bb.update_layout(
                title=f"Bollinger Bands ({bb_period}-day, {bb_std}σ)",
                xaxis_title="Date",
                yaxis_title="Price",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=50, b=0)
            )

            st.plotly_chart(fig_bb, use_container_width=True)

            # Bollinger Band interpretation
            current_price = price.iloc[-1] if not price.empty else None
            current_upper_bb = upper_bb.iloc[-1] if not upper_bb.empty else None
            current_lower_bb = lower_bb.iloc[-1] if not lower_bb.empty else None

            if all(v is not None for v in [current_price, current_upper_bb, current_lower_bb]):
                if current_price > current_upper_bb:
                    st.markdown(f"""
                    <div style="padding:10px; border-radius:5px; background-color:{COLORS["RED"]}; color:white;">
                        <strong>Bollinger Bands: Overbought</strong> - Price is above the upper band, indicating potential overbought conditions.
                    </div>
                    """, unsafe_allow_html=True)
                elif current_price < current_lower_bb:
                    st.markdown(f"""
                    <div style="padding:10px; border-radius:5px; background-color:{COLORS["GREEN"]}; color:white;">
                        <strong>Bollinger Bands: Oversold</strong> - Price is below the lower band, indicating potential oversold conditions.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="padding:10px; border-radius:5px; background-color:{COLORS["NEUTRAL"]}; color:white;">
                        <strong>Bollinger Bands: Neutral</strong> - Price is within the bands, indicating normal trading conditions.
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning(f"Insufficient price data for {ticker} to perform volatility analysis.")

    with tech_tabs[3]:  # Volume Analysis
        st.markdown("### Volume Analysis")

        if not volume.empty and not price.empty:
            # Calculate volume moving average
            volume_ma = volume.rolling(window=20).mean()

            # Create volume figure
            fig_vol = go.Figure()

            # Add volume bars
            fig_vol.add_trace(go.Bar(
                x=volume.index,
                y=volume,
                name="Volume",
                marker_color=COLORS["BLUE"]
            ))

            # Add volume moving average
            fig_vol.add_trace(go.Scatter(
                x=volume_ma.index,
                y=volume_ma,
                name="20-day MA",
                line=dict(color=COLORS["RED"])
            ))

            # Update layout
            fig_vol.update_layout(
                title="Volume Analysis",
                xaxis_title="Date",
                yaxis_title="Volume",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=50, b=0)
            )

            st.plotly_chart(fig_vol, use_container_width=True)

            # Volume and Price Relationship
            st.markdown("#### Volume and Price Relationship")

            # Calculate price change
            price_change = price.pct_change()

            # Create scatter plot of volume vs price change
            fig_vol_price = px.scatter(
                x=price_change,
                y=volume,
                title="Volume vs. Price Change",
                labels={"x": "Daily Price Change (%)", "y": "Volume"},
                color=price_change,
                color_continuous_scale=["red", "green"],
                opacity=0.7
            )

            # Update layout
            fig_vol_price.update_layout(
                xaxis_tickformat=".2%",
                margin=dict(l=0, r=0, t=50, b=0)
            )

            st.plotly_chart(fig_vol_price, use_container_width=True)

            # Volume interpretation
            current_volume = volume.iloc[-1] if not volume.empty else None
            avg_volume = volume.mean() if not volume.empty else None

            if current_volume is not None and avg_volume is not None:
                volume_ratio = current_volume / avg_volume

                if volume_ratio > 1.5:
                    st.markdown(f"""
                    <div style="padding:10px; border-radius:5px; background-color:{COLORS["BLUE"]}; color:white;">
                        <strong>Volume: High</strong> - Current volume ({current_volume:,.0f}) is {volume_ratio:.1f}x the average, indicating strong interest.
                    </div>
                    """, unsafe_allow_html=True)
                elif volume_ratio < 0.5:
                    st.markdown(f"""
                    <div style="padding:10px; border-radius:5px; background-color:{COLORS["ORANGE"]}; color:white;">
                        <strong>Volume: Low</strong> - Current volume ({current_volume:,.0f}) is {volume_ratio:.1f}x the average, indicating weak interest.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="padding:10px; border-radius:5px; background-color:{COLORS["NEUTRAL"]}; color:white;">
                        <strong>Volume: Normal</strong> - Current volume ({current_volume:,.0f}) is close to the average.
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning(f"Insufficient volume data for {ticker} to perform volume analysis.")
