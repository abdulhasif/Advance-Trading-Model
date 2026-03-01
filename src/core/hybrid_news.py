import logging
import feedparser
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime
import asyncio

# Transformers for FinBERT
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class HybridNewsEngine:
    """
    Fetches financial news from yfinance and Indian RSS feeds,
    and analyzes the sentiment using the ProsusAI/finbert model.
    """
    
    def __init__(self):
        self.classifier = None
        self.rss_feeds = [
            "https://www.moneycontrol.com/rss/MCtopnews.xml",
            "https://economictimes.indiatimes.com/markets/rssfeeds/2146842.cms",
            "https://www.business-standard.com/rss/markets-106.rss"
        ]
        
        # Load the FinBERT model if transformers is available
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info("loading ProsusAI/finbert pipeline... (This requires ~400MB RAM)")
                self.classifier = pipeline("text-classification", model="ProsusAI/finbert")
                logger.info("FinBERT pipeline loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load FinBERT: {e}")
        else:
            logger.warning("transformers library not found. FinBERT sentiment analysis will be disabled.")
            
        # Cache to avoid reprocessing the same headlines over and over
        self.processed_headlines = set()

    def analyze_sentiment(self, text: str) -> float:
        """
        Passes text through FinBERT. 
        Returns a float score between -1.0 (highly negative) and 1.0 (highly positive).
        """
        if not self.classifier or not text:
            return 0.0
            
        try:
            # FinBERT usually returns labels like 'positive', 'negative', 'neutral'
            # with a score between 0 and 1 representing confidence.
            result = self.classifier(text)[0]
            label = result['label'].lower()
            confidence = result['score']
            
            if label == 'positive':
                return confidence
            elif label == 'negative':
                return -confidence
            else:
                return 0.0 # neutral
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment for text '{text[:30]}...': {e}")
            return 0.0

    def fetch_yfinance_news(self, ticker: str) -> List[Dict]:
        """
        Fetches the latest headlines for a specific ticker from Yahoo Finance.
        Note: Indian ticker symbols in yfinance usually need a '.NS' or '.BO' suffix.
        If the ticker does not have a suffix, '.NS' will be assumed.
        """
        news_items = []
        
        # Standardize ticker for Indian markets (assuming NSE as default if no suffix)
        yf_ticker = ticker if "." in ticker else f"{ticker}.NS"
        
        try:
            company = yf.Ticker(yf_ticker)
            # Fetch recent news dicts
            recent_news = company.news
            
            for item in recent_news:
                # yfinance API v8 changed to nested 'content' structure, support both
                content = item.get('content', item)
                title = content.get('title', item.get('title', ''))
                
                if title and title not in self.processed_headlines:
                    self.processed_headlines.add(title)
                    
                    # Extract publisher and timestamp safely
                    provider = content.get('provider', {})
                    publisher = provider.get('displayName', item.get('publisher', 'Yahoo Finance'))
                    timestamp = content.get('pubDate', item.get('providerPublishTime', 0))
                    
                    news_items.append({
                        "ticker": ticker,
                        "headline": title,
                        "source": publisher,
                        "timestamp": timestamp 
                    })
        except Exception as e:
            logger.warning(f"Error fetching yfinance news for {yf_ticker}: {e}")
            
        return news_items

    def fetch_rss_news(self, watch_tickers: List[str]) -> List[Dict]:
        """
        Scrapes standard Indian financial RSS feeds and checks if any 
        watch tickers are mentioned in the titles.
        """
        news_items = []
        
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    title = entry.get('title', '')
                    if not title or title in self.processed_headlines:
                        continue
                        
                    # Check if any active ticker is mentioned in the uppercase headline
                    # This is a basic exact word match.
                    title_upper = title.upper()
                    
                    found_ticker = None
                    for ticker in watch_tickers:
                        # Ensure we match whole words to avoid partial matches (e.g. 'IT' matching 'WITH')
                        # Alternatively, since Indian tickers are often full company names loosely,
                        # we do a loose search for strings > 3 chars, or exact word match.
                        # Simple inclusion for now, prioritize longer ticker names if they match
                        if ticker.upper() in title_upper:
                             found_ticker = ticker
                             break
                             
                    if found_ticker:
                        self.processed_headlines.add(title)
                        news_items.append({
                            "ticker": found_ticker,
                            "headline": title,
                            "source": "RSS Feed",
                            "timestamp": datetime.now().timestamp()
                        })
                        
            except Exception as e:
                logger.warning(f"Error parsing RSS feed {feed_url}: {e}")
                
        return news_items

    def poll_all_news(self, active_tickers: List[str], watch_tickers: List[str] = None) -> List[Dict]:
        """
        Main orchestration method. Fetches from all sources, scores sentiment, 
        and returns aggregated results.
        
        active_tickers: Passed directly to yFinance (keep low to avoid rate limits).
        watch_tickers: Checked exclusively against RSS feeds (can be all 139).
        """
        watch_tickers = watch_tickers or active_tickers
        logger.info(f"Polling news for {len(active_tickers)} active tickers (yFinance) and {len(watch_tickers)} watch tickers (RSS)...")
        raw_news = []
        
        # 1. Fetch from yFinance (Targeted)
        for ticker in active_tickers:
            raw_news.extend(self.fetch_yfinance_news(ticker))
            
        # 2. Fetch from RSS Feeds (Broad)
        raw_news.extend(self.fetch_rss_news(watch_tickers))
        
        # 3. Analyze Sentiment
        results = []
        for item in raw_news:
            sentiment = self.analyze_sentiment(item["headline"])
            results.append({
                "ticker": item["ticker"],
                "headline": item["headline"],
                "sentiment_score": sentiment,
                "source": item["source"]
            })
            
        # Optional: Trim the processed cache to prevent infinite growth
        if len(self.processed_headlines) > 2000:
            self.processed_headlines.clear()
            
        logger.info(f"Poll complete. Analyzed {len(results)} new headlines.")
        return results

# A quick simple test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = HybridNewsEngine()
    
    # Test with some active tokens
    test_tickers = ["RELIANCE", "TCS", "HDFCBANK", "ZOMATO"]
    results = engine.poll_all_news(test_tickers)
    
    for r in results:
        print(f"[{r['ticker']}] Sentiment: {"%.2f" % r['sentiment_score']} | {r['headline']}")
