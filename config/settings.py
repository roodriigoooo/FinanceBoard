"""
Configuration settings for the Finance Board application.
"""

# Application settings
APP_TITLE = "Enhanced Stock Metrics Dashboard"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Default values
DEFAULT_TICKER = "AAPL"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_TIMEFRAME = "3M"

# Color scheme
COLORS = {
    "BLUE": "#3274a1",
    "ORANGE": "#e1813c",
    "NEUTRAL": "#909090",
    "GREEN": "#2ca02c",
    "RED": "#d62728",
    "PURPLE": "#9467bd",
    "YELLOW": "#ffed6f"
}

# Custom CSS
CUSTOM_CSS = f"""
[data-testid="stMetric"] div:first-child {{font-size:0.85rem;}}
[data-testid="stMetricValue"] {{font-weight:600;}}
[data-testid="stMetricDeltaPositive"] {{color:{COLORS["BLUE"]} !important;}}
[data-testid="stMetricDeltaNegative"] {{color:{COLORS["ORANGE"]} !important;}}
.insight-box {{padding:10px;border-radius:5px;background:#f0f2f6;margin-bottom:12px;}}
footer {{visibility:hidden;}}
"""

# Tab configuration
DEFAULT_TABS = {
    "tech_tab_toggle": True,
    "fund_tab_toggle": True,
    "news_tab_toggle": True
}

# Risk control defaults
DEFAULT_MAX_POSITION_SIZE = 20  # percentage
