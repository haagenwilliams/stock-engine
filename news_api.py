import os
import os
import requests
from datetime import datetime, timedelta

class NewsAPI:
    def __init__(self):
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        self.base_url = 'https://www.alphavantage.co/query'

    def get_news(self, symbol):
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.api_key,
                'limit': 10
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'feed' not in data:
                return []
            
            news_items = []
            for item in data['feed'][:10]:  # Limit to 10 items
                news_items.append({
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'time_published': item.get('time_published', ''),
                    'summary': item.get('summary', '')
                })
            
            return news_items
            
        except Exception as e:
            print(f'Error fetching news: {str(e)}')
            return []