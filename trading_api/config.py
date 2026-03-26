from trading_core.core.config.base_config import *

# ─────────────────────────────────────────────────────────────────────────────
# 10. UI & DASHBOARD AESTHETICS
# ─────────────────────────────────────────────────────────────────────────────
JET_THEME_PRIMARY     = "#00f2ff"
JET_THEME_SECONDARY   = "#7000ff"
DASHBOARD_REFRESH_SEC = 30
STATE_WRITE_INTERVAL  = 1.0

# Market Regime Viz
REGIME_WINDOW         = 40
REGIME_MIN_SIGNALS    = 10
REGIME_BIAS_TRENDING  = 60
REGIME_BIAS_VOLATILE  = 40
REGIME_CONV_TRENDING  = 60
REGIME_CONV_VOLATILE  = 45

# News & Sentiment
SENTIMENT_THRESHOLD   = 0.5
NEWS_POLL_INTERVAL    = 300
NEWS_CACHE_LIMIT      = 2000
NEWS_RSS_FEEDS        = [
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/2146842.cms",
    "https://www.business-standard.com/rss/markets-106.rss"
]
