#!/usr/bin/env python3
import io
import re
import glob
import json
import textblob
import os.path, time
import datetime as dt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as vad
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

################# Tweet Processing Methods #################

def create_tweets_df(file_paths):
    read_in_tweets = []

    counter = 0

    for file_path in file_paths:
        if (counter > 5):
            break
        counter = counter + 1

        # Load in raw tweet data
        print("Loading in {}".format(file_path))
        with open(file_path, 'r', encoding="utf8") as json_file:
            tweets = json.load(json_file)
            file_name_tokens = file_path.split("_")
            for tweet in tweets:
                tweet['date'] = file_name_tokens[3]
            read_in_tweets.extend(tweets)

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
                    tweet['user']['verified'],
                    tweet['date']
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
                                     'user_verified',
                                     'date'
                                    ])

    # Process hashtags into a list
    tweet_df['hashtags'] = tweet_df['hashtags'].apply(lambda x: [d['text'].lower() for d in x])

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

def tb_sentiment_analysis(tweet_df):
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

    tweet_df['sentiment_score'] = sentiment_score
    tweet_df['subjectivity'] = subjectivity
    tweet_df['overall_sentiment'] = sentiment

    return tweet_df

def vader_sentiment_analysis(tweet_df):
    sentiment = vad()
    # Making additional columns for sentiment score in the vader dataframe
    sen = ['Positive','Negative','Neutral']
    sentiments = [sentiment.polarity_scores(i) for i in tweet_df['text'].values]
    tweet_df['negative_score'] = [i['neg'] for i in sentiments]
    tweet_df['positive_score'] = [i['pos'] for i in sentiments]
    tweet_df['neutral_score'] = [i['neu'] for i in sentiments]
    tweet_df['compound_score'] = [i['compound'] for i in sentiments]
    score = tweet_df['compound_score'].values
    t = []
    for i in score:
        if i >= 0.05 :
            t.append('Positive')
        elif i <= -0.05 :
            t.append('Negative')
        else:
            t.append('Neutral')
    tweet_df['overall_sentiment'] = t

    return tweet_df

#############################################################

########### Visualization and Statistics Methods ############

def print_tb_stats(stats_file, df, vaccine_name):
    stats_file.write("{} Textblob statistics:\n".format(vaccine_name))
    avg_sent = df['sentiment_score'].mean()
    ovr_sent = 'Neutral'
    stats_file.write("\t > Average {} Sentiment Score: {}\n".format(vaccine_name, avg_sent))
    if avg_sent < 0:
        ovr_sent = 'Negative'
    elif avg_sent > 0:
        ovr_sent = 'Positive'
    else:
        ovr_sent = 'Neutral'
    stats_file.write("\t > Overall {} Sentiment: {}\n\n".format(vaccine_name, ovr_sent))

def print_vader_stats(stats_file, df, vaccine_name):
    stats_file.write("{} Vader statistics:\n".format(vaccine_name))
    avg_sent = df['sentiment_score'].mean()
    ovr_sent = 'Neutral'
    stats_file.write("\t > Average {} Sentiment Score: {}\n".format(vaccine_name, avg_sent))
    if avg_sent < 0:
        ovr_sent = 'Negative'
    elif avg_sent > 0:
        ovr_sent = 'Positive'
    else:
        ovr_sent = 'Neutral'
    stats_file.write("\t > Overall {} Sentiment: {}\n\n".format(vaccine_name, ovr_sent))

def overall_sentiments_visualization(file_path, df):
    print(file_path)
    sents = ["Negative", "Neutral", "Positive"]
    sent_colors = ["#800000", "#696969", "#006400"]
    plt.figure()
    plt.figure(figsize=(20, 15))
    plot = sns.countplot(data = df, x = 'overall_sentiment', order = sents, palette = sent_colors)
    plot.figure.savefig(file_path)

def subjectivity_visualization(file_path, df):
    print(file_path)
    plt.figure()
    plt.figure(figsize=(20, 15))
    plot = sns.displot(data = df, x = 'negative_score', bins = 10)
    plot.savefig(file_path)
    plt.close("all")

def sentiments_visualization(file_path, df):
    print(file_path)
    sent_colors = ["#800000", "#696969", "#006400"]
    plt.figure()
    plt.figure(figsize=(20, 15))
    plot = sns.displot(data = df, x = 'compound_score', bins = 20, palette = sent_colors, hue = 'overall_sentiment')
    plot.savefig(file_path)
    plt.close("all")

def sentiments_date_visualization(file_path, df):
    print(file_path)
    sents = ["Negative", "Neutral", "Positive"]
    sent_colors = ["#800000", "#696969", "#006400"]
    plt.figure()
    plt.figure(figsize=(20, 15))
    plot = sns.countplot(data = df, x = 'date', hue_order = sents, palette = sent_colors, hue = 'overall_sentiment')
    plot.figure.savefig(file_path)

#############################################################

if __name__ == '__main__':
    file_paths = glob.glob("./Active Data/*.json")
    
    print("Creating tweets dataframe...")
    tweet_df = create_tweets_df(file_paths)
    tweet_df.drop_duplicates(subset='text', inplace=True)
    print("[SUCCESS] Created tweet dataframe containing all {} tweets.\n".format(len(tweet_df.index)))

    print("Processing tweet text data...")
    tweet_df['text'] = tweet_df['text'].apply(preprocess_tweet)
    print("[SUCCESS] Succesfully processed tweet text for all tweets.\n")

    print("Applying TextBlob sentiment analysis...")
    #tweet_df = sentiment_analysis(tweet_df)
    tweet_df = vader_sentiment_analysis(tweet_df)
    print("[SUCCESS] Sentiment analysis complete for all tweets.\n")

    print("Creating vaccine type data segmentations...")
    
    print("AstraZeneca segmentation...")
    astrazeneca_keywords = ["astrazeneca", "astrazenecavaccine"]
    astrazeneca_df = pd.DataFrame()
    for keyword in astrazeneca_keywords:
        astrazeneca_df = astrazeneca_df.append(tweet_df.loc[tweet_df["text"].str.contains(keyword, case=False)])
    astrazeneca_df = astrazeneca_df.append(tweet_df[tweet_df.hashtags.apply(lambda x: any(item for item in astrazeneca_keywords if item in x))])
    astrazeneca_df = astrazeneca_df.loc[astrazeneca_df.astype(str).drop_duplicates().index]

    print("Pfizer segmentation...")
    pfizer_keywords = ["pfizer", "pfizervaccine"]
    pfizer_df = pd.DataFrame()
    for keyword in pfizer_keywords:
        pfizer_df = pfizer_df.append(tweet_df.loc[tweet_df["text"].str.contains(keyword, case=False)])
    pfizer_df = pfizer_df.append(tweet_df[tweet_df.hashtags.apply(lambda x: any(item for item in astrazeneca_keywords if item in x))])
    pfizer_df = pfizer_df.loc[pfizer_df.astype(str).drop_duplicates().index]

    print("Johnson and Johnson segmentation...")
    jj_keywords = ["johnsonandjohnson", "johnsonandjohnsonvaccine", "johnson and johnson"]
    jj_df = pd.DataFrame()
    for keyword in jj_keywords:
        jj_df = jj_df.append(tweet_df.loc[tweet_df["text"].str.contains(keyword, case=False)])
    jj_df = jj_df.append(tweet_df[tweet_df.hashtags.apply(lambda x: any(item for item in astrazeneca_keywords if item in x))])
    jj_df = jj_df.loc[jj_df.astype(str).drop_duplicates().index]

    print("Moderna segmentation...")
    moderna_keywords = ["moderna", "modernavaccine"]
    moderna_df = pd.DataFrame()
    for keyword in moderna_keywords:
        moderna_df = moderna_df.append(tweet_df.loc[tweet_df["text"].str.contains(keyword, case=False)])
    moderna_df = moderna_df.append(tweet_df[tweet_df.hashtags.apply(lambda x: any(item for item in astrazeneca_keywords if item in x))])
    moderna_df = moderna_df.loc[moderna_df.astype(str).drop_duplicates().index]

    print("Sputnik segmentation...")
    sputnik_keywords = ["sputnik", "sputnikvaccine"]
    sputnik_df = pd.DataFrame()
    for keyword in sputnik_keywords:
        sputnik_df = sputnik_df.append(tweet_df.loc[tweet_df["text"].str.contains(keyword, case=False)])
    sputnik_df = sputnik_df.append(tweet_df[tweet_df.hashtags.apply(lambda x: any(item for item in astrazeneca_keywords if item in x))])
    sputnik_df = sputnik_df.loc[sputnik_df.astype(str).drop_duplicates().index]
    print("[SUCCESS] Succesfully created all vaccine type segmentations.\n")


    print("Generating statistics..")
    with open('textblob_sentiment_analysis.txt', 'a') as tb_stat_file:
        print_tb_stats(tb_stat_file, tweet_df, "All")
        print_tb_stats(tb_stat_file, astrazeneca_df, "AstraZeneca")
        print_tb_stats(tb_stat_file, pfizer_df, "Pfizer")
        print_tb_stats(tb_stat_file, jj_df, "Johnson and Johnson")
        print_tb_stats(tb_stat_file, moderna_df, "Moderna")
        print_tb_stats(tb_stat_file, sputnik_df, "Sputnik")
    


    print("Generating visualizations..")
    # All tweets visualizations
    all_overall_sentiments_path = "./Visualizations/all_overall_sentiments.png"
    overall_sentiments_visualization(all_overall_sentiments_path, tweet_df)
    all_subjectivity_path = "./Visualizations/all_subjectivity.png"
    subjectivity_visualization(all_subjectivity_path, tweet_df)
    all_sentiments_path = "./Visualizations/all_sentiments.png"
    sentiments_visualization(all_sentiments_path, tweet_df)
    all_sentiments_date_path = "./Visualizations/all_date_sentiments.png"
    sentiments_date_visualization(all_sentiments_date_path, tweet_df)

    # AstraZeneca visualizations
    astrazeneca_overall_sentiments_path = "./Visualizations/astrazeneca_overall_sentiments.png"
    overall_sentiments_visualization(astrazeneca_overall_sentiments_path, astrazeneca_df)
    astrazeneca_subjectivity_path = "./Visualizations/astrazeneca_subjectivity.png"
    subjectivity_visualization(astrazeneca_subjectivity_path, astrazeneca_df)
    astrazeneca_sentiments_path = "./Visualizations/astrazeneca_sentiments.png"
    sentiments_visualization(astrazeneca_sentiments_path, astrazeneca_df)
    astrazeneca_sentiments_date_path = "./Visualizations/astrazeneca_date_sentiments.png"
    sentiments_date_visualization(astrazeneca_sentiments_date_path, astrazeneca_df)

    # Pfizer visualizations
    pfizer_overall_sentiments_path = "./Visualizations/pfizer_overall_sentiments.png"
    overall_sentiments_visualization(pfizer_overall_sentiments_path, pfizer_df)
    pfizer_subjectivity_path = "./Visualizations/pfizer_subjectivity.png"
    subjectivity_visualization(pfizer_subjectivity_path, pfizer_df)
    pfizer_sentiments_path = "./Visualizations/pfizer_sentiments.png"
    sentiments_visualization(pfizer_sentiments_path, pfizer_df)
    pfizer_sentiments_date_path = "./Visualizations/pfizer_date_sentiments.png"
    sentiments_date_visualization(pfizer_sentiments_date_path, pfizer_df)

    # Johnson and Johnson visualizations
    jj_overall_sentiments_path = "./Visualizations/jj_overall_sentiments.png"
    overall_sentiments_visualization(jj_overall_sentiments_path, jj_df)
    jj_subjectivity_path = "./Visualizations/jj_subjectivity.png"
    subjectivity_visualization(jj_subjectivity_path, jj_df)
    jj_sentiments_path = "./Visualizations/jj_sentiments.png"
    sentiments_visualization(jj_sentiments_path, jj_df)
    jj_sentiments_date_path = "./Visualizations/jj_date_sentiments.png"
    sentiments_date_visualization(jj_sentiments_date_path, jj_df)

    # Moderna visualizations
    moderna_overall_sentiments_path = "./Visualizations/moderna_overall_sentiments.png"
    overall_sentiments_visualization(moderna_overall_sentiments_path, moderna_df)
    moderna_subjectivity_path = "./Visualizations/moderna_subjectivity.png"
    subjectivity_visualization(moderna_subjectivity_path, moderna_df)
    moderna_sentiments_path = "./Visualizations/moderna_sentiments.png"
    sentiments_visualization(moderna_sentiments_path, moderna_df)
    moderna_sentiments_date_path = "./Visualizations/moderna_date_sentiments.png"
    sentiments_date_visualization(moderna_sentiments_date_path, moderna_df)

    # Sputnik visualizations
    sputnik_overall_sentiments_path = "./Visualizations/sputnik_overall_sentiments.png"
    overall_sentiments_visualization(sputnik_overall_sentiments_path, sputnik_df)
    sputnik_subjectivity_path = "./Visualizations/sputnik_subjectivity.png"
    subjectivity_visualization(sputnik_subjectivity_path, sputnik_df)
    sputnik_sentiments_path = "./Visualizations/sputnik_sentiments.png"
    sentiments_visualization(sputnik_sentiments_path, sputnik_df)
    sputnik_sentiments_date_path = "./Visualizations/sputnik_date_sentiments.png"
    sentiments_date_visualization(sputnik_sentiments_date_path, sputnik_df)
    
    print("[SUCCESS] Generated all visualizations.\n")
