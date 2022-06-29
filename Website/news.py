from newsapi import NewsApiClient 
import pandas as pd
import datetime
import yfinance as yf

choice='AAPL'
data = yf.download(tickers=choice, period = '2y')
#data_for_today = data.iloc[0]
data.to_csv('apple_stock_data.csv')

API_KEY = '58aee54bbd21427ba01d9131648bba70'
news_api = NewsApiClient(API_KEY)

today = datetime.date.today().strftime("%Y-%m-%d") 
top_headlines = news_api.get_everything(q='apple',
                                    from_param=today,
                                    to=today,
                                    sort_by='relevancy',
                                    language='en')

if top_headlines['status']=="ok":
    news = top_headlines['articles']
    news_headlines = []
    for details in news:
        news_headlines.append(details['title'])
dataset = pd.DataFrame(data = {'headlines':news_headlines[:20]})
dataset.to_csv('today_news.csv')