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
nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer as vad
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

################# Tweet Processing Methods #################

def create_tweets_df(file_paths):
    read_in_tweets = []

    for file_path in file_paths:
        # Load in raw tweet data
        print("Loading in {}".format(file_path))
        with open(file_path, "r", encoding="utf8") as json_file:
            tweets = json.load(json_file)
            file_name_tokens = file_path.split("_")
            for tweet in tweets:
                tweet["date"] = file_name_tokens[3]
            read_in_tweets.extend(tweets)

    # Extract the required data
    raw_tweets = read_in_tweets
    tweet_data = [[
                    tweet["id"],
                    tweet["text"],
                    tweet["favorite_count"],
                    tweet["retweet_count"],
                    tweet["entities"]["hashtags"],
                    tweet["entities"]["user_mentions"],
                    tweet["user"]["name"],
                    tweet["user"]["screen_name"],
                    tweet["user"]["description"],
                    tweet["user"]["statuses_count"],
                    tweet["user"]["location"],
                    tweet["user"]["followers_count"],
                    tweet["user"]["verified"],
                    tweet["date"]
                  ] for tweet in raw_tweets]

    # Create dataframe
    tweet_df = pd.DataFrame(data=tweet_data, 
                            columns=["id",
                                     "text",
                                     "favorite_count",
                                     "retweet_count",
                                     "hashtags",
                                     "user_mentions",
                                     "user_name",
                                     "user_screen_name",
                                     "user_description",
                                     "user_statuses_count",
                                     "user_location",
                                     "user_followers_count",
                                     "user_verified",
                                     "date"
                                    ])

    # Process hashtags into a list
    tweet_df["hashtags"] = tweet_df["hashtags"].apply(lambda x: [d["text"].lower() for d in x])

    return tweet_df

def preprocess_tweet(txt):
    # Remove mentions
    txt = re.sub(r"@[A-Za-z0-9_]+", "", txt)
    # Remove hashtags
    txt = re.sub(r"#", "", txt)
    # Remove retweets:
    txt = re.sub(r"RT : ", "", txt)
    # Remove urls
    txt = re.sub(r"https?:\/\/[A-Za-z0-9\.\/]+", "", txt)
    # Removes stop words
    txt = txt.lower()
    txt = re.sub(r"rt ", "", txt)
    text_tokens = word_tokenize(txt)
    [word for word in text_tokens if not word in stopwords.words("english")]
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words("english")]
    txt = (" ").join(tokens_without_sw)
    txt = txt.encode("ascii", "ignore")
    txt = txt.decode()
    return txt.strip()

def tb_sentiment_analysis(tweet_df):
    sentiment = []
    subjectivity = []
    sentiment_score = []

    sentiment_data = tweet_df[["text"]]

    for text in sentiment_data["text"].values:
        tweet = textblob.TextBlob(text)
        sentiment_score.append(tweet.sentiment[0])
        if tweet.sentiment[0] < 0:
            sentiment.append("Negative")
        elif tweet.sentiment[0] > 0:
            sentiment.append("Positive")
        else:
            sentiment.append("Neutral")
        subjectivity.append(tweet.sentiment[1])

    tweet_df["tb_sentiment_score"] = sentiment_score
    tweet_df["tb_subjectivity"] = subjectivity
    tweet_df["tb_overall_sentiment"] = sentiment

    return tweet_df

def vader_sentiment_analysis(tweet_df):
    sentiment = vad()
    # Making additional columns for sentiment score in the vader dataframe
    sen = ["Positive","Negative","Neutral"]
    sentiments = [sentiment.polarity_scores(i) for i in tweet_df["text"].values]
    tweet_df["vader_negative_score"] = [i["neg"] for i in sentiments]
    tweet_df["vader_positive_score"] = [i["pos"] for i in sentiments]
    tweet_df["vader_neutral_score"] = [i["neu"] for i in sentiments]
    tweet_df["vader_compound_score"] = [i["compound"] for i in sentiments]
    score = tweet_df["vader_compound_score"].values
    t = []
    for i in score:
        if i >= 0.05 :
            t.append("Positive")
        elif i <= -0.05 :
            t.append("Negative")
        else:
            t.append("Neutral")
    tweet_df["vader_overall_sentiment"] = t

    return tweet_df

#############################################################

########### Visualization and Statistics Methods ############

def print_tb_stats(stats_file, df, vaccine_name):
    stats_file.write("{} Textblob statistics:\n".format(vaccine_name))
    avg_sent = df["tb_sentiment_score"].mean()
    avg_subj = df["tb_subjectivity"].mean()
    ovr_sent = "Neutral"
    stats_file.write("Average {} Sentiment Score: {}\n".format(vaccine_name, avg_sent))
    stats_file.write("Average {} Subjectivity: {}\n".format(vaccine_name, avg_subj))
    if avg_sent < 0:
        ovr_sent = "Negative"
    elif avg_sent > 0:
        ovr_sent = "Positive"
    else:
        ovr_sent = "Neutral"
    stats_file.write("Overall {} Sentiment: {}\n\n".format(vaccine_name, ovr_sent))

def print_vader_stats(stats_file, df, vaccine_name):
    stats_file.write("{} Vader statistics:\n".format(vaccine_name))
    avg_neg = df["vader_negative_score"].mean()
    avg_neut = df["vader_neutral_score"].mean()
    avg_pos = df["vader_positive_score"].mean()
    avg_score = df["vader_compound_score"].mean()
    ovr_sent = "Neutral"
    stats_file.write("Average {} Negative Score: {}\n".format(vaccine_name, avg_neg))
    stats_file.write("Average {} Neutral Score: {}\n".format(vaccine_name, avg_neut))
    stats_file.write("Average {} Positive Score: {}\n".format(vaccine_name, avg_pos))
    if avg_score >= 0.05 :
        ovr_sent = "Positive"
    elif avg_score <= -0.05 :
        ovr_sent = "Negative"
    else:
        ovr_sent = "Neutral"
    stats_file.write("Overall {} Sentiment: {}\n\n".format(vaccine_name, ovr_sent))

## Textblob Visualizations

def tb_overall_sentiments_visualization(file_path, df):
    print(file_path)
    sents = ["Negative", "Neutral", "Positive"]
    sent_colors = ["#800000", "#696969", "#006400"]
    plt.figure()
    plt.figure(figsize = (20, 10))
    plot = sns.countplot(data = df, x = "tb_overall_sentiment", order = sents, palette = sent_colors)
    plot.figure.savefig(file_path)
    plt.close("all")

def tb_sentiments_visualization(file_path, df):
    print(file_path)
    sents = ["Negative", "Neutral", "Positive"]
    sent_colors = ["#800000", "#696969", "#006400"]
    plt.figure()
    plt.figure(figsize = (20, 10))
    plot = sns.displot(data = df, x = "tb_sentiment_score", bins = 20, palette = sent_colors, hue = "tb_overall_sentiment", hue_order = sents)
    plot.savefig(file_path)
    plt.close("all")

def tb_subjectivity_visualization(file_path, df):
    print(file_path)
    plt.figure()
    plt.figure(figsize = (20, 10))
    plot = sns.displot(data = df, x = "tb_subjectivity", bins = 10)
    plot.savefig(file_path)
    plt.close("all")

def tb_sentiments_date_visualization(file_path, df):
    print(file_path)
    sents = ["Negative", "Neutral", "Positive"]
    sent_colors = ["#800000", "#696969", "#006400"]
    plt.figure()
    plt.figure(figsize = (20, 10))
    plot = sns.countplot(data = df, x = "date", hue_order = sents, palette = sent_colors, hue = "tb_overall_sentiment")
    plot.figure.savefig(file_path)
    plt.close("all")

## Vader Visualizations

def vader_overall_sentiments_visualization(file_path, df):
    print(file_path)
    sents = ["Negative", "Neutral", "Positive"]
    sent_colors = ["#800000", "#696969", "#006400"]
    plt.figure()
    plt.figure(figsize = (20, 10))
    plot = sns.countplot(data = df, x = "vader_overall_sentiment", order = sents, palette = sent_colors)
    plot.figure.savefig(file_path)
    plt.close("all")

def vader_sentiments_visualization(file_path, df):
    print(file_path)
    sents = ["Negative", "Neutral", "Positive"]
    sent_colors = ["#800000", "#696969", "#006400"]
    plt.figure()
    plt.figure(figsize = (20, 10))
    plot = sns.displot(data = df, x = "vader_compound_score", bins = 20, palette = sent_colors, hue = "vader_overall_sentiment", hue_order = sents)
    plot.savefig(file_path)
    plt.close("all")

def vader_sentiments_date_visualization(file_path, df):
    print(file_path)
    sents = ["Negative", "Neutral", "Positive"]
    sent_colors = ["#800000", "#696969", "#006400"]
    plt.figure()
    plt.figure(figsize = (20, 10))
    plot = sns.countplot(data = df, x = "date", hue_order = sents, palette = sent_colors, hue = "vader_overall_sentiment")
    plot.figure.savefig(file_path)
    plt.close("all")

def generate_all_visualizations(df, segment_name):
    # Tweets Textblob visualizations
    tb_overall_sentiments_path = "./Visualizations/Textblob/{}_overall_sentiments.png".format(segment_name)
    tb_overall_sentiments_visualization(tb_overall_sentiments_path, df)
    tb_subjectivity_path = "./Visualizations/Textblob/{}_subjectivity.png".format(segment_name)
    tb_subjectivity_visualization(tb_subjectivity_path, df)
    tb_sentiments_path = "./Visualizations/Textblob/{}_sentiments.png".format(segment_name)
    tb_sentiments_visualization(tb_sentiments_path, df)
    tb_sentiments_date_path = "./Visualizations/Textblob/{}_date_sentiments.png".format(segment_name)
    tb_sentiments_date_visualization(tb_sentiments_date_path, df)
    # Tweets Vader visualizations
    vader_overall_sentiments_path = "./Visualizations/Vader/{}_overall_sentiments.png".format(segment_name)
    vader_overall_sentiments_visualization(vader_overall_sentiments_path, df)
    vader_sentiments_path = "./Visualizations/Vader/{}_sentiments.png".format(segment_name)
    vader_sentiments_visualization(vader_sentiments_path, df)
    vader_sentiments_date_path = "./Visualizations/Vader/{}_date_sentiments.png".format(segment_name)
    vader_sentiments_date_visualization(vader_sentiments_date_path, df)

#############################################################

if __name__ == "__main__":
    file_paths = glob.glob("./Active Data/*.json")
    
    print("Creating tweets dataframe...")
    tweet_df = create_tweets_df(file_paths)
    tweet_df.drop_duplicates(subset="text", inplace=True)
    print("[SUCCESS] Created tweet dataframe containing all {} tweets.\n".format(len(tweet_df.index)))

    print("Processing tweet text data...")
    tweet_df["text"] = tweet_df["text"].apply(preprocess_tweet)
    print("[SUCCESS] Succesfully processed tweet text for all tweets.\n")

    print("Applying TextBlob sentiment analysis...")
    tweet_df = tb_sentiment_analysis(tweet_df)
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
    # Textblob Stats
    textblob_stat_file_path = "./textblob_sentiment_analysis.txt"
    with open("textblob_sentiment_analysis.txt", "w") as tb_stat_file:
        print_tb_stats(tb_stat_file, tweet_df, "All")
        print_tb_stats(tb_stat_file, astrazeneca_df, "AstraZeneca")
        print_tb_stats(tb_stat_file, pfizer_df, "Pfizer")
        print_tb_stats(tb_stat_file, jj_df, "Johnson and Johnson")
        print_tb_stats(tb_stat_file, moderna_df, "Moderna")
        print_tb_stats(tb_stat_file, sputnik_df, "Sputnik")
    print("[SUCCESS] Succesfully generated and wrote Textblob statistics to \"./{}\".\n".format(textblob_stat_file_path))
    # Vader Stats
    vader_stat_file_path = "./vader_sentiment_analysis.txt"
    with open(vader_stat_file_path, "w") as vader_stat_file:
        print_vader_stats(vader_stat_file, tweet_df, "All")
        print_vader_stats(vader_stat_file, astrazeneca_df, "AstraZeneca")
        print_vader_stats(vader_stat_file, pfizer_df, "Pfizer")
        print_vader_stats(vader_stat_file, jj_df, "Johnson and Johnson")
        print_vader_stats(vader_stat_file, moderna_df, "Moderna")
        print_vader_stats(vader_stat_file, sputnik_df, "Sputnik")
    print("[SUCCESS] Succesfully generated and wrote Vader statistics to \"./{}\".\n".format(vader_stat_file_path))
    
    print("Generating visualizations..")
    generate_all_visualizations(tweet_df, "All")
    generate_all_visualizations(astrazeneca_df, "AstraZeneca")
    generate_all_visualizations(pfizer_df, "Pfizer")
    generate_all_visualizations(jj_df, "JohnsonandJohnson")
    generate_all_visualizations(moderna_df, "Moderna")
    generate_all_visualizations(sputnik_df, "Sputnik")

    print("[SUCCESS] Generated all visualizations.\n")
