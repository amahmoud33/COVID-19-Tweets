#!/usr/bin/env python3
import re
import io
import csv
import tweepy
import json
from datetime import datetime

keywords = ["covid+vaccination",
            "covid+vaccine",
            "covid-19+vaccine",
            "vaccine+for+covid",
            "pfizer+vaccine",
            "pfizer+covid",
            "johnson+and+johnson+covid",
            "johnson+and+johnson+vaccine",
            "moderna+vaccine",
            "moderna+covid",
            "astrazeneca+vaccine"
            "astrazeneca+covid",
            "sputnik+vaccine",
            "sputnik+covid",
            "china+vaccine",
            "vaccine+sick",
            "vaccine+trust",
            "vaccine+allergy",
            "covid-19+vaccination",
            "vaccination+for+covid",
            "vaccination+sick",
            "vaccination+trust",
            "vaccination+allergy"]

lang_en = "en"

def query_covid_keywords(queries, language = "en", count = 200):
    raw_tweets = []

    # Make the API call for the tweet queries
    for query in queries:
        raw_tweets = raw_tweets + api.search(query, lang = language, count = count)
    
    print("Successfully gathered " + str(len(raw_tweets)) + " tweets.")

    # Collect the JSON response from each tweet
    with open("./Raw Data/covid-19_vaccine_tweets_%s.json" % datetime.now().strftime("%d-%m-%Y_%H-%M-%S"), "w") as f:
        tweets = []
        for raw_tweet in raw_tweets:
            tweets.append(raw_tweet._json)
        # Dumping the JSON from the tweets
        json.dump(tweets, f)

# NOTE: YOU MUST ADD TWITTER DEV AUTH KEYS
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# Setting access token and secret
auth.set_access_token(access_token, access_token_secret)
# Creating the API object while passing in auth information
api = tweepy.API(auth)

query_covid_keywords(keywords, lang_en, 10000)