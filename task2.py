from logging import exception
from flask import get_template_attribute
from pytz import UTC
from tweepy import Client
from datetime import datetime
from datetime import timedelta
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from emot.emo_unicode import UNICODE_EMOJI

nltk.download('vader_lexicon')

BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAIHFbAEAAAAA0oBVG5orLLErnyAqw2po3fOae5w%3D4lgZWoMOyGG496F2aNACoKOdDCaDnxqret6oFPLToE244O6Tx6"





def get_time_step():
    start = datetime(2020, 1, 1, hour=0,minute=0, tzinfo=UTC)
    half_year = timedelta(days=128)
    next = start + half_year

    time_step =[start]
    current = start
    while current < datetime.now(start.tzinfo):
        current += half_year
        time_step.append(current)
    return time_step
    # print(time_step)



fake_tweet = "New Harris County #Houston #COVID19 numbers online tonight @HoustonChron. Forget the 2 week grace period you heard this AM at the House Energy Commerce testimony (which didn't make sense to me), we're at this point now, unfortunately.  Discussing tomorrow @NewDay @CNN early start https://t.co/oAqEbXbsEC"
def hash_tags_analysis(text):
    tags = {tag.strip("#") for tag in text.split() if tag.startswith("#")}
    return tags
    
def at_analysis(text):
    tags = {tag.strip("@") for tag in text.split() if tag.startswith("@")}
    return tags    

# print(hash_tags_analysis(fake_tweet))
# print(at_analysis(fake_tweet))


text1 = '''
    Hilarious 😂. The feeling of making a sale 😎, The feeling of actually fulfilling orders 😒
    '''
def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    def convert_emojis(text):
        for emot in UNICODE_EMOJI:
            text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
        return text.replace("_"," ")
    text = convert_emojis(text)
    return sia.polarity_scores(text)

# print(sentiment_analysis(text1))


# TODO task 2 working in progress
def topic():
    topics = pd.DataFrame({'datetime':[], 'domain':[]})
    #topics = pd.DataFrame([[]], columns=['datetime', 'domain'])
    id = 1243807211815649285
    client = Client(BEARER_TOKEN)
    tw = client.get_tweet(id, tweet_fields=["context_annotations", "created_at","entities","public_metrics"], user_fields="public_metrics")
    # uid = tw.data.author_id
    # print(tw)
    # tags = {tag.strip("#") for tag in text.split() if tag.startswith("#")}
    #print(tags)
    #print(tw.data.context_annotations)
    # dti = pd.date_range(start="2019-12-31", periods=6, freq="6M")
    # dti = dti.tz_localize("UTC")
    #print(dti) 
    sent_date = tw.data.created_at
    ts = get_time_step()
    #print(sent_date)
    # print(sent_date in range(ts[0],ts[1]))
    # print(sent_date>ts[0])
    # print(tw.data.entities)
    # print(tw.data.public_metrics)
    domains = {d["domain"]["name"] for d in tw.data.context_annotations}
    for d in domains:
        temp = pd.DataFrame([[sent_date, d]], columns=['datetime', 'domain'])
        topics = pd.concat([topics, temp])
    print(topics)
    topics.set_index('datetime')
    # for i in tw.data:
    #     print(i)
    # print(uid)
    # print(tw.includes["users"][0].public_metrics)

id = 1243807211815649285

# run for each tweet, need add if statement depends on rumour or not
def get_tweet(id):
    client = Client(BEARER_TOKEN)
    tweet = client.get_tweet(id, expansions = ["author_id"],tweet_fields=["context_annotations", "created_at","entities","public_metrics"], user_fields="public_metrics")
    text = tweet.data.text
    print(text)
    
    # task 1
    domains = {d["domain"]["name"] for d in tweet.data.context_annotations}
    print(domains)
    
    # task 2 TODO still working on group over time 
    
    # task 3
    print(hash_tags_analysis(text))
    # task 4
    print(sentiment_analysis(text))
    # task 5
    pm = tweet.includes["users"][0].public_metrics
    print(pm)
    
    # extra 1 at analysis
    print(at_analysis(text))

get_tweet(id)

# topic()