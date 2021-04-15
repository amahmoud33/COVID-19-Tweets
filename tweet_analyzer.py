#!/usr/bin/env python3
import io
import re
import glob
import json
import textblob
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def create_tweets_df(file_paths):
    read_in_tweets = []

    for file_path in file_paths:
        # Load in raw tweet data
        print("Loading in {}".format(file_path))
        with open(file_path, 'r', encoding="utf8") as json_file:
            read_in_tweets.extend(json.load(json_file))
    
    # Extract the required data
    raw_tweets = read_in_tweets
    tweet_data = [[
                    tweet['id'],
                    tweet['text'],
                    tweet['favorite_count'],
                    tweet['retweet_count'],
                    tweet['entities']['hashtags'],
                    tweet['entities']['user_mentions'],
                    tweet['user']['name'],
                    tweet['user']['screen_name'],
                    tweet['user']['description'],
                    tweet['user']['statuses_count'],
                    tweet['user']['location'],
                    tweet['user']['followers_count'],
                    tweet['user']['verified']
                  ] for tweet in raw_tweets]

    # Create dataframe
    tweet_df = pd.DataFrame(data=tweet_data, 
                            columns=['id', 
                                     'text',
                                     'favorite_count',
                                     'retweet_count',
                                     'hashtags',
                                     'user_mentions',
                                     'user_name',
                                     'user_screen_name',
                                     'user_description',
                                     'user_statuses_count',
                                     'user_location',
                                     'user_followers_count',
                                     'user_verified'
                                    ])
    return tweet_df

def preprocess_tweet(txt):
    # Remove mentions
    txt = re.sub(r'@[A-Za-z0-9_]+', '', txt)
    # Remove hashtags
    txt = re.sub(r'#', '', txt)
    # Remove retweets:
    txt = re.sub(r'RT : ', '', txt)
    # Remove urls
    txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', txt)
    # Removes stop words
    txt = txt.lower()
    txt = re.sub(r'rt ', '', txt)
    text_tokens = word_tokenize(txt)
    [word for word in text_tokens if not word in stopwords.words('english')]
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]
    txt = (" ").join(tokens_without_sw)
    txt = txt.encode("ascii", "ignore")
    txt = txt.decode()
    return txt.strip()

def sentiment_analysis(tweet_df):
    sentiment = []
    subjectivity = []
    sentiment_score = []

    sentiment_data = tweet_df[['text']]

    for text in sentiment_data['text'].values:
        tweet = textblob.TextBlob(text)
        sentiment_score.append(tweet.sentiment[0])
        if tweet.sentiment[0] < 0:
            sentiment.append('Negative')
        elif tweet.sentiment[0] > 0:
            sentiment.append('Positive')
        else:
            sentiment.append('Neutral')
        subjectivity.append(tweet.sentiment[1])

    tweet_df['Sentiment Score'] = sentiment_score
    tweet_df['Subjectivity'] = subjectivity
    tweet_df['Overall Sentiment'] = sentiment

    return tweet_df

if __name__ == '__main__':
    file_paths = glob.glob("./Active Data/*.json")
    
    print("Creating tweets dataframe...")
    tweet_df = create_tweets_df(file_paths)

    tweet_df.drop_duplicates(subset='text', inplace=True)
    print("Processing tweet text data...")
    tweet_df['text'] = tweet_df['text'].apply(preprocess_tweet)

    print("Applying TextBlob sentiment analysis...")
    tweet_df = sentiment_analysis(tweet_df)

    print("[SUCCESS] Sentiment for tweet data analyzed!")
    print(tweet_df)
    avg_sent = tweet_df['Sentiment Score'].mean()
    ovr_sent = 'Neutral'
    print("Average Sentiment Score: ", avg_sent)
    if avg_sent < 0:
        ovr_sent = 'Negative'
    elif avg_sent > 0:
        ovr_sent = 'Positive'
    else:
        ovr_sent = 'Neutral'
    print("Overall Sentiment: ", ovr_sent)

    tweet_df.to_csv(r'tweet_df.csv')

