from matplotlib.pyplot import title
from spacy import Vocab
from flask import Flask, render_template
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objs as go
import yfinance
import datetime
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import nltk
from datetime import date

try:
    nltk.data.find('tokenizers/punkt')
except :
    nltk.download('all')

filename = 'newss_classification_nb.sav'
model = load_model('dense.h5')
nb_clf = pickle.load(open(filename, 'rb'))

my_stop_words = text.ENGLISH_STOP_WORDS

def pipeline(train_data,nb_clf):
    stemmer = PorterStemmer()
    def stem_words(text):
        return " ".join([stemmer.stem(word) for word in text.split()])
    train_data["headlines"] = train_data["headlines"].apply(lambda text: stem_words(text))
    count_vectorizer =  CountVectorizer(stop_words= my_stop_words, max_features= 77)
    feature_vector =  count_vectorizer.fit(train_data.headlines)
    train_ds_features =  count_vectorizer.transform(train_data.headlines)
    predictedSentiment = nb_clf.predict(train_ds_features.toarray())
    return predictedSentiment

app = Flask(__name__)

@app.route('/analytics')
def blah():
    data = pd.read_csv('apple_stock_data.csv')
    data = data.set_index('Date')

    fig = go.Figure()
    fig.add_trace(go.Line(x=data.index,y=data['Open'], name='Opening Price'))
    fig.add_trace(go.Line(x=data.index,y=data['High'], name='High Price'))
    fig.add_trace(go.Line(x=data.index,y=data['Low'], name='Low Price'))
    fig.add_trace(go.Line(x=data.index,y=data['Close'], name='Close Price'))
    fig.update_layout(title = "Stock Price for Apple")
    fig.update_layout(dragmode=False)
    graphJSON_maingraph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('analysis.html', graphJSON=graphJSON_maingraph)

@app.route('/')
def notdash():

    data = pd.read_csv('apple_stock_data.csv')
    blah = data[['Date','Open', 'Close', 'Volume']]
    data = data.set_index('Date')

    #fig = go.Figure()
    #fig.add_trace(go.Line(x=data.index,y=data['Open'], name='Opening Price'))
    #fig.add_trace(go.Line(x=data.index,y=data['High'], name='High Price'))
    #fig.add_trace(go.Line(x=data.index,y=data['Low'], name='Low Price'))
    #fig.add_trace(go.Line(x=data.index,y=data['Close'], name='Close Price'))
    #fig.update_layout(title = "Stock Price for Apple")
    #fig.update_layout(dragmode=False)
    #graphJSON_maingraph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    news = pd.read_csv('today_news.csv')
    extra = pd.read_csv('today_news.csv')

    scalerfile = 'scalers/scaler_close.sav'
    scaler_close = pickle.load(open(scalerfile, 'rb'))

    scalerfile = 'scalers/scaler_open.sav'
    scaler_open = pickle.load(open(scalerfile, 'rb'))

    scalerfile = 'scalers/scaler_volume.sav'
    scaler_volume = pickle.load(open(scalerfile, 'rb'))
    
    X = data.drop(['High', 'Low', 'Adj Close'],axis=1).values[-1:]
    close, opening, volume = X[0, 0], X[0, 1], X[0, 2]
    X[0:1, 0:1] = scaler_close.transform(X[0:1, 0:1])
    X[0:1, 1:2] = scaler_open.transform(X[0:1, 1:2])
    X[0:1, 2:3] = scaler_volume.transform(X[0:1, 2:3])

    news_sentiment = pipeline(news, nb_clf)
    #fig = ""
    count = 0
    pos = 0
    neg = 0
    for i in news_sentiment:
        if int(i)==1:
            pos+=1
        else:
            neg+=1
        count+=1
    data = pd.DataFrame({'Sentiment': ['Positive', 'Negative'], 'Count': [pos, neg]})
    pos = int(pos/count*100)
    neg = int(neg/count*100)
    print(pos, neg)
    mean_score = np.mean(news_sentiment)
    print(data)    
    #fig = px.bar(data, x='Sentiment',y='Count')
    #fig.write_image("static/images/bar_graph.png")
    #graphJSON_bar = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    today = date.today()
    today = today.strftime("%d/%m/%Y")
    news_sentiment = np.reshape([np.mean(news_sentiment)], (-1, 1))
    X = np.concatenate([news_sentiment, X], axis=1)
    prediction = model.predict(X)[0][0]
    if int(prediction)==1:
        prediction = "The stock price might increase tomorrow"
    else:
        prediction = "The stock price will decrease tomorrow"
    print(prediction)

    return render_template('index.html', today_date=today, close=close, opening=opening, volume=volume,news_headlines=extra['headlines'].to_list(), prediction=prediction, positive=pos, negative=neg, sentiment_score=mean_score, dates = blah['Date'].apply(str).to_list(), opening_price = blah['Open'].to_list(), volume_list = blah['Volume'].to_list())

