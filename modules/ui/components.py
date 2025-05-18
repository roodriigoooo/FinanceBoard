"""
UI components for the Finance Board application.
"""

from datetime import date, timedelta
import streamlit as st
import pandas as pd

from config.settings import COLORS, DEFAULT_TICKER, DEFAULT_BENCHMARK, DEFAULT_TIMEFRAME, DEFAULT_TABS


def setup_page_config():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="Enhanced Stock Metrics Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply custom CSS
    st.markdown(f"""
    <style>
    /* Global styles - white background and black text */
    .stApp {{
        background-color: white;
        color: black;
    }}

    /* Hide the module list in the sidebar */
    [data-testid="stSidebarNav"] {{display: none !important;}}

    /* Improved metrics styling */
    [data-testid="stMetric"] {{
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        color: black;
    }}
    [data-testid="stMetric"] div:first-child {{
        font-size: 0.9rem;
        font-weight: 500;
        color: black;
    }}
    [data-testid="stMetricValue"] {{
        font-weight: 700;
        font-size: 1.2rem !important;
        color: black;
    }}
    [data-testid="stMetricDeltaPositive"] {{color:{COLORS["GREEN"]} !important;}}
    [data-testid="stMetricDeltaNegative"] {{color:{COLORS["RED"]} !important;}}

    /* Custom font for the entire app */
    html, body, [class*="css"], .stMarkdown, p, div {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: black;
    }}

    /* Improved headers */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        color: black;
    }}

    /* Improved containers */
    .insight-box {{
        padding: 15px;
        border-radius: 8px;
        background: white;
        margin-bottom: 15px;
        border-left: 4px solid {COLORS["BLUE"]};
        color: black;
    }}

    /* Hide footer */
    footer {{visibility: hidden;}}

    /* Improved sidebar */
    [data-testid="stSidebar"] {{
        background-color: white;
        border-right: 1px solid #eee;
        color: black;
    }}

    /* Improved buttons */
    .stButton > button {{
        border-radius: 6px;
        font-weight: 500;
        background-color: white;
        color: black;
    }}

    /* Improved tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
        color: black;
    }}

    /* Ensure text inputs have black text */
    .stTextInput input, .stNumberInput input, .stDateInput input {{
        color: black !important;
    }}

    /* Ensure selectbox text is black */
    .stSelectbox div[data-baseweb="select"] span {{
        color: black !important;
    }}

    /* Ensure all text in expanders is black */
    .streamlit-expanderHeader, .streamlit-expanderContent {{
        color: black !important;
    }}
    </style>""", unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with all controls."""
    with st.sidebar:
        ticker, benchmark = render_asset_selection()
        st.divider()
        start_date, end_date = render_timeframe_selection()
        st.divider()
        render_risk_alerts()

    return ticker, benchmark, start_date, end_date


def render_asset_selection():
    """Render the asset selection section in the sidebar."""
    st.header("🔍 Asset & Universe")
    ticker = st.text_input("Search Ticker", DEFAULT_TICKER,
                          help="Enter a stock ticker (e.g., AAPL, MSFT, GOOGL).").upper().strip()
    benchmark = st.text_input("Benchmark (optional)", DEFAULT_BENCHMARK,
                             help="Enter a benchmark ticker (e.g., SPY, QQQ).").upper().strip()

    # Initialize session state for watchlist and alerts if they don't exist
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []
    if 'price_alerts' not in st.session_state:
        st.session_state.price_alerts = {}

    st.subheader("Watchlist Management")
    if ticker:  # Ensure ticker has a value
        add_button_key = f"add_watchlist_{ticker}"
        remove_button_key = f"remove_watchlist_{ticker}"
        if ticker not in st.session_state.watchlist:
            if st.button(f"➕ Add {ticker} to Watchlist", key=add_button_key, use_container_width=True):
                if ticker not in st.session_state.watchlist:  # Double check
                    st.session_state.watchlist.append(ticker)
                st.rerun()
        else:
            if st.button(f"➖ Remove {ticker} from Watchlist", key=remove_button_key, use_container_width=True):
                if ticker in st.session_state.watchlist:
                    st.session_state.watchlist.remove(ticker)
                st.rerun()

    if st.session_state.watchlist:
        st.caption("Current Watchlist:")
        for item_wl in st.session_state.watchlist[:]:
            st.markdown(f"• {item_wl}")
    else:
        st.caption("No assets in watchlist.")

    return ticker, benchmark


def render_timeframe_selection():
    """Render the timeframe selection section in the sidebar."""
    st.header("📅 Timeframe & Analysis")
    st.subheader("Time Range")

    today = date.today()

    quick_range_options = ("1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "Max")

    # Get current or default index for quick_range_selector
    quick_range_idx = quick_range_options.index(DEFAULT_TIMEFRAME)  # Default
    if "quick_range_selector" in st.session_state:
        try:
            quick_range_idx = quick_range_options.index(st.session_state.quick_range_selector)
        except ValueError:
            quick_range_idx = quick_range_options.index(DEFAULT_TIMEFRAME)  # Fallback

    current_quick_range = st.selectbox(
        "Preset Range",
        quick_range_options,
        index=quick_range_idx,
        key="quick_range_selector",
        help="Select a preset time range for the analysis.",
    )

    # Calculate start date based on selected range
    if current_quick_range == "1D":
        start_default = today - timedelta(days=1)
    elif current_quick_range == "5D":
        start_default = today - timedelta(days=5)
    elif current_quick_range == "1M":
        start_default = today - timedelta(days=30)
    elif current_quick_range == "3M":
        start_default = today - timedelta(days=90)
    elif current_quick_range == "6M":
        start_default = today - timedelta(days=180)
    elif current_quick_range == "YTD":
        start_default = date(today.year, 1, 1)
    elif current_quick_range == "1Y":
        start_default = today - timedelta(days=365)
    elif current_quick_range == "3Y":
        start_default = today - timedelta(days=365 * 3)
    elif current_quick_range == "5Y":
        start_default = today - timedelta(days=365 * 5)
    else:  # Max
        start_default = today - timedelta(days=365 * 10)

    # Initialize start_date and end_date with defaults from quick_range_selector
    start_date = start_default
    end_date = today

    with st.expander("Custom Date Range", expanded=False):
        # These will update start_date and end_date if the user interacts
        start_date_custom = st.date_input("Start date", start_date,
                                         key="custom_start_date_input")  # Use current start_date as default
        end_date_custom = st.date_input("End date", end_date,
                                       key="custom_end_date_input")  # Use current end_date as default

        if start_date_custom > end_date_custom:
            st.error("Custom start date must be before custom end date.")
            # Keep previous valid dates (start_date, end_date from quick_range or last valid custom)
        else:
            start_date = start_date_custom
            end_date = end_date_custom

    st.subheader("Analysis Toggles")

    # First, initialize the session state if needed
    if "show_tech_tab" not in st.session_state:
        st.session_state.show_tech_tab = DEFAULT_TABS["tech_tab_toggle"]
    if "show_fund_tab" not in st.session_state:
        st.session_state.show_fund_tab = DEFAULT_TABS["fund_tab_toggle"]
    if "show_news_tab" not in st.session_state:
        st.session_state.show_news_tab = DEFAULT_TABS["news_tab_toggle"]

    # Define callback functions to update session state
    def update_tech_tab():
        st.session_state.tech_tab_toggle = st.session_state.show_tech_checkbox

    def update_fund_tab():
        st.session_state.fund_tab_toggle = st.session_state.show_fund_checkbox

    def update_news_tab():
        st.session_state.news_tab_toggle = st.session_state.show_news_checkbox

    # Create the checkboxes with callback functions
    st.checkbox("Show Technical Analysis Tab",
               value=st.session_state.show_tech_tab,
               key="show_tech_checkbox",
               on_change=update_tech_tab)

    st.checkbox("Show Fundamental Analysis Tab",
               value=st.session_state.show_fund_tab,
               key="show_fund_checkbox",
               on_change=update_fund_tab)

    st.checkbox("Show News Tab",
               value=st.session_state.show_news_tab,
               key="show_news_checkbox",
               on_change=update_news_tab)

    # Update our tracking variables for next time
    st.session_state.show_tech_tab = st.session_state.get("show_tech_checkbox", st.session_state.show_tech_tab)
    st.session_state.show_fund_tab = st.session_state.get("show_fund_checkbox", st.session_state.show_fund_tab)
    st.session_state.show_news_tab = st.session_state.get("show_news_checkbox", st.session_state.show_news_tab)

    return start_date, end_date


def render_risk_alerts():
    """Render the risk and alerts section in the sidebar."""
    st.header("⚙️ Risk & Alerts")
    st.subheader("Risk Controls (Illustrative)")
    # This slider updates the session state directly through its key
    st.slider(
        "Max Position Size (% of Portfolio)",
        min_value=0, max_value=100, value=20, step=5,
        key="max_pos_size_slider_ctrl",
        help="Illustrative: maximum percentage one asset can occupy in a portfolio."
    )

    st.subheader("Price Alerts")
    # Get the current ticker from session state
    ticker = st.session_state.get("ticker", "")

    # Use a consistent key for the number_input regardless of ticker to preserve its state better during reruns
    alert_price_target_input_key = "alert_price_target_input_field"

    # Determine current target for the input field
    current_alert_target_for_input = None
    if ticker and ticker in st.session_state.price_alerts and st.session_state.price_alerts[ticker].get("active"):
        current_alert_target_for_input = st.session_state.price_alerts[ticker].get("target")

    alert_price_target = st.number_input(
        f"Set alert for {ticker or 'selected asset'} at price:",
        min_value=0.01,
        value=current_alert_target_for_input,  # Pre-fill if active alert exists for current ticker
        step=0.01,
        format="%.2f",
        key=alert_price_target_input_key,  # Consistent key
        help="Enter a target price. An active alert for the current ticker will be shown below."
    )

    col_alert1, col_alert2 = st.columns(2)
    with col_alert1:
        # Button key should be dynamic if its action depends on the ticker
        set_alert_button_key = f"set_alert_btn_{ticker}" if ticker else "set_alert_btn_no_ticker"
        if st.button("Set/Update Alert", key=set_alert_button_key, use_container_width=True):
            if ticker and alert_price_target is not None and alert_price_target > 0:
                st.session_state.price_alerts[ticker] = {"target": alert_price_target, "active": True}
                st.success(f"Alert for {ticker} set/updated to ${alert_price_target:.2f}.")
                st.rerun()  # Rerun to update UI, including pre-filling the input if needed
            elif not ticker:
                st.warning("Enter a ticker to set an alert.")
            else:  # alert_price_target is None or <=0
                st.warning("Enter a valid positive price target.")

    with col_alert2:
        if ticker and ticker in st.session_state.price_alerts and st.session_state.price_alerts[ticker].get("active"):
            clear_alert_button_key = f"clear_alert_btn_{ticker}"
            if st.button("Clear Alert", key=clear_alert_button_key, use_container_width=True):
                st.session_state.price_alerts[ticker]["active"] = False
                # Optionally remove the ticker from price_alerts if no longer needed: del st.session_state.price_alerts[ticker]
                st.info(f"Alert for {ticker} cleared.")
                st.rerun()  # Rerun to update UI

    if ticker and ticker in st.session_state.price_alerts and st.session_state.price_alerts[ticker].get("active"):
        alert_info = st.session_state.price_alerts[ticker]
        st.caption(f"ℹ️ Active alert for {ticker}: Target ${alert_info['target']:.2f}")


def render_top_bar(ticker, asset_name, price, volume):
    """Render the top bar with key metrics."""
    top_bar_cols = st.columns([3, 1.5, 1.5, 1.5, 0.5])  # Adjusted column ratios

    with top_bar_cols[0]:
        st.subheader(f"{asset_name} ({ticker})")

    with top_bar_cols[1]:
        latest_price = price.iloc[-1] if not price.empty else None
        st.metric("Latest Price", f"${latest_price:,.2f}" if latest_price is not None else "–")

    with top_bar_cols[2]:
        if len(price) >= 2:
            price_change_pct = (price.iloc[-1] / price.iloc[-2] - 1)
            price_change_abs = price.iloc[-1] - price.iloc[-2]
            st.metric(
                "Change",
                f"{price_change_abs:,.2f} ({price_change_pct:.2%})" if not pd.isna(price_change_pct) else "–",
                delta=f"{price_change_pct:.2%}" if not pd.isna(price_change_pct) else None,
                delta_color="normal"
            )
        else:
            st.metric("Change", "–")

    with top_bar_cols[3]:
        latest_volume = volume.iloc[-1] if not volume.empty else None
        st.metric("Volume", f"{latest_volume:,.0f}" if latest_volume is not None else "–")

    with top_bar_cols[4]:
        st.markdown(
            """
            <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
                <span title="For educational purposes only. Not financial advice. Data may be delayed or inaccurate.">⚠️</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")  # Horizontal line after top bar
