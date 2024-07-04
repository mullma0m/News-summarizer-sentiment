import streamlit as st
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, date, timedelta
from newspaper import Article
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API keys from environment variables
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
NEWSAPI_API_KEY = os.getenv('NEWSAPI_API_KEY')

# Function to get news articles based on the ticker symbol and date range from FinnHub
def get_news_finnhub(ticker, start_date, end_date):
    url = f"https://finnhub.io/api/v1/company-news"
    params = {
        'symbol': ticker,
        'from': start_date,
        'to': end_date,
        'token': FINNHUB_API_KEY
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return [{'api_source': 'FinnHub', 'actual_source': article['source'], **article} for article in response.json()]

# Function to get news articles based on the ticker symbol and date range from NewsAPI
def get_news_newsapi(ticker, start_date, end_date):
    url = f"https://newsapi.org/v2/everything"
    params = {
        'q': ticker,
        'from': start_date,
        'to': end_date,
        'sortBy': 'publishedAt',
        'language': 'en',  
        'apiKey': NEWSAPI_API_KEY
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    articles = response.json().get('articles', [])
    for article in articles:
        article['source'] = article['source']['name']
    return [{'api_source': 'NewsAPI', 'actual_source': article['source'], **article} for article in articles]

# Function to map sentiment labels to a unified set of labels as defined previously
def map_sentiment(label, model_name):
    model_to_sentiments = {
        'distilbert-base-uncased-finetuned-sst-2-english': {'POSITIVE': 'Positive', 'NEGATIVE': 'Negative'},
        'finiteautomata/bertweet-base-sentiment-analysis': {'POS': 'Positive', 'NEG': 'Negative', 'NEU': 'Neutral'},
        'cardiffnlp/twitter-roberta-base-sentiment': {'LABEL_2': 'Positive', 'LABEL_0': 'Negative', 'LABEL_1': 'Neutral'},
        'nlptown/bert-base-multilingual-uncased-sentiment': {
            '1 star': 'Negative', '2 stars': 'Negative', '3 stars': 'Neutral', 
            '4 stars': 'Positive', '5 stars': 'Positive'
        },
        'Mulla88/group2_fin_model': {'LABEL_2': 'Positive', 'LABEL_0': 'Negative', 'LABEL_1': 'Neutral'}
    }
    
    # Default return label if the model or label mapping does not exist
    return model_to_sentiments.get(model_name, {}).get(label, label)

# Configure logging
logging.basicConfig(level=logging.ERROR, filename='error.log')

# Streamlit app layout
st.set_page_config(page_title="Ticker News Summarizer", layout="wide")
st.title('Ticker News Summarizer and Sentiment Analyzer')

# User inputs
st.header("Enter Details")
ticker = st.text_input('Enter a ticker symbol (e.g. AAPL, LCID, BTC, ETH):')
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input('Start date', value=date.today() - timedelta(days=7))
with col2:
    end_date = st.date_input('End date', value=date.today())

# Model selection
summarization_model = st.selectbox('Choose a summarization model', [
    'facebook/bart-large-cnn', 
    't5-base', 
    'google/pegasus-xsum', 
    'distilbart-cnn-12-6',
    't5-small'
])
sentiment_model = st.selectbox('Choose a sentiment analysis model', [
    'distilbert-base-uncased-finetuned-sst-2-english',
    'finiteautomata/bertweet-base-sentiment-analysis',
    'cardiffnlp/twitter-roberta-base-sentiment',
    'nlptown/bert-base-multilingual-uncased-sentiment',
    'Mulla88/group2_fin_model'
])

# Customizable parameters
max_length = st.slider('Maximum summary length', 50, 300, 200)
min_length = st.slider('Minimum summary length', 20, 150, 80)

# Buttons to manage fetching news
col3, col4 = st.columns([1, 1])
with col3:
    get_news_button = st.button('Get News')
with col4:
    stop_fetch = st.button('Stop')

if 'fetching_news' not in st.session_state:
    st.session_state.fetching_news = False

if stop_fetch:
    st.session_state.fetching_news = False

if get_news_button and ticker:
    st.session_state.fetching_news = True

if st.session_state.fetching_news and ticker:
    with st.spinner('Fetching news...'):
        try:
            finnhub_news = get_news_finnhub(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            newsapi_news = get_news_newsapi(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            # Combine both news sources
            all_news = finnhub_news + newsapi_news
            
            # Convert all dates to a standard format and sort by date
            for article in all_news:
                if 'datetime' in article:
                    article['date'] = datetime.fromtimestamp(article['datetime'])
                else:
                    article['date'] = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
            
            all_news = sorted(all_news, key=lambda x: x['date'], reverse=True)

            if all_news:
                st.header("News Articles")
                for article in all_news:
                    if not st.session_state.fetching_news:
                        break
                    news_url = article.get('url')
                    title = article.get('headline') or article.get('title')
                    published_at = article['date']
                    api_source = article.get('api_source')
                    actual_source = article.get('actual_source')

                    if news_url and title:
                        try:
                            news_article = Article(news_url)
                            news_article.download()
                            news_article.parse()
                            
                            # Summarize and analyze sentiment
                            summarizer = pipeline("summarization", model=summarization_model)
                            summary = summarizer(news_article.text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
                            
                            if sentiment_model == 'Mulla88/group2_fin_model':
                                custom_model_name = 'Mulla88/group2_fin_model'
                                custom_tokenizer = AutoTokenizer.from_pretrained(custom_model_name)
                                custom_model = AutoModelForSequenceClassification.from_pretrained(custom_model_name)
                                sentiment_analyzer = pipeline("sentiment-analysis", model=custom_model, tokenizer=custom_tokenizer)
                            else:
                                sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model)
                                
                            sentiment_label = sentiment_analyzer(summary)[0]['label']
                            unified_sentiment = map_sentiment(sentiment_label, sentiment_model)
                            
                            # Display the article information
                            st.subheader(f"**Title:** {title}")
                            st.write(f"(*API Source: {api_source}*, *Actual Source: {actual_source}*)")
                            st.write(f"**Date:** {published_at.strftime('%Y-%m-%d %H:%M:%S')}")
                            st.write(f"[Read full article]({news_url})")
                            st.write(f"**Summary:** {summary}")
                            st.write(f"**Sentiment:** {unified_sentiment}")
                            st.write("---")
                        except Exception as e:
                            logging.error(f"Could not process article: {news_url}. Error: {e}")
                            continue
            else:
                st.write("No news articles found for the given date range.")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
else:
    st.write("Please enter a ticker symbol.")