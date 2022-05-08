from logging import exception
import string
from flask import get_template_attribute
import numpy as np
from pytz import UTC
import scipy
from tweepy import Client
from datetime import datetime, timedelta
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from emot.emo_unicode import UNICODE_EMOJI
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import keras


nltk.download('vader_lexicon')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

file = "covid.data.txt"


BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAIHFbAEAAAAA0oBVG5orLLErnyAqw2po3fOae5w%3D4lgZWoMOyGG496F2aNACoKOdDCaDnxqret6oFPLToE244O6Tx6"





def get_time_step():
    time_step = []
    for y in range(2020, 2023, 1):
        for m in [1, 6]:
            time_step.append(datetime(y, m, 1, hour=0,minute=0, tzinfo=UTC))
    return time_step

time_step = get_time_step()


# fake_tweet = "New Harris County #Houston #COVID19 numbers online tonight @HoustonChron. Forget the 2 week grace period you heard this AM at the House Energy Commerce testimony (which didn't make sense to me), we're at this point now, unfortunately.  Discussing tomorrow @NewDay @CNN early start https://t.co/oAqEbXbsEC"
def hash_tags_analysis(text):
    tags = {tag.strip("#") for tag in text.split() if tag.startswith("#")}
    return tags
    
def at_analysis(text):
    tags = {tag.strip("@") for tag in text.split() if tag.startswith("@")}
    return tags    

# print(hash_tags_analysis(fake_tweet))
# print(at_analysis(fake_tweet))


# text1 = '''Hilarious 😂. The feeling of making a sale 😎, The feeling of actually fulfilling orders 😒'''

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
    

def get_time_slot(test_date):
    for i in range(0,len(time_step) - 1):
        if test_date >= time_step[i] and test_date<=time_step[i+1]:
            return i

def tokenize_tweet(string_data:str):
    wordnet_lemmatizer = WordNetLemmatizer()
    tokenized = nltk.RegexpTokenizer('\w+')
    data = string_data.replace('\n', '')
    data = data.lower()
    data = re.sub('https?://\S+|www\.\S+', '', data)
    data = re.sub('[%s]' % re.escape(string.punctuation), '', data)
    data = tokenized.tokenize(data)
    data = [i for i in data if i not in stopwords]
    data = [wordnet_lemmatizer.lemmatize(word) for word in data]
    data = ' '.join(data)   
    return data


topics = [{}, {}]
hashtags = [{}, {}]
sentiment = [dict({'neg': [], 'neu': [], 'pos': [], 'compound': []}),
             dict({'neg': [], 'neu': [], 'pos': [], 'compound': []})]
public_metrics = [dict({'followers_count': [], 'following_count': [], 'tweet_count': [], 'listed_count': []}),
                  dict({'followers_count': [], 'following_count': [], 'tweet_count': [], 'listed_count': []})]
count = [0, 0]
count_time = [0,0,0,0,0]
ats = [{}, {}]
topic_time = [{},{},{},{},{}]

# run for each tweet, need add if statement depends on rumour or not
def get_tweet(id, model, vectorizer):
    client = Client(BEARER_TOKEN, wait_on_rate_limit=True)
    tweet = client.get_tweet(id, expansions = ["author_id"],tweet_fields=["context_annotations", "created_at","entities","public_metrics"], user_fields=["verified", "public_metrics"])
    if tweet.data == None:
        return
    text = tweet.data.text
    # print(text)
    varified = tweet.includes["users"][0].verified
    # print(varified)
    test_text = tokenize_tweet(text)
    # print(test_text)
    testtfidf = vectorizer.transform([test_text])
    # print(testtfidf)
    testtfidf = scipy.sparse.csr_matrix(testtfidf).todense()
    # print(testtfidf)
    # test = tf.constant([[testtfidf, int(varified)]])
    # print(test)
    testpd = pd.DataFrame(testtfidf)
    test_verified = pd.DataFrame({"verified":[int(varified)]})
    test = pd.concat([testpd, test_verified], axis=1)
    test = tf.convert_to_tensor(test)
    # print(test)

    prediction = model.predict(test)
    prediction = (prediction > 0.5).astype("int32")
    prediction = np.ndarray.flatten(prediction)[0]
    # print(prediction)
    # 0 is non rumour 
    count[prediction] += 1
    
    # print(tweet.data.context_annotations)
    # task 1 
    # What are the topics of COVID-19 rumours, and how do they differ from the non-rumours?
    domains = {d["domain"]["name"] for d in tweet.data.context_annotations}
    
    for d in domains:
        topics[prediction][d] = topics[prediction].get(d, 0) + 1

    # print(domains)
    # task 2 
    # How do COVID-19 rumour topics or trends evolve over time?
    if prediction:
        sent_date = tweet.data.created_at
        slot = get_time_slot(sent_date)
        count_time[slot] += 1
        for d in domains:
            topic_time[slot][d] = topic_time[slot].get(d, 0) + 1
    # print(sent_date)
    # task 3
    # What are the popular hashtags of COVID-19 rumours and non-rumours? How much overlap or difference do they share?
    tags = hash_tags_analysis(text)
    for t in tags:
        hashtags[prediction][t] = hashtags[prediction].get(t, 0) + 1
    # task 4
    # Do rumour source tweets convey a different sentiment/emotion to the non-rumour source tweets? What about their replies? 
    sentiment_score = sentiment_analysis(text)
    # sentiment[prediction] = {sentiment[prediction][k].append(sentiment_score[k]) for k in sentiment_score.keys()}
    
    for k in sentiment_score.keys():
        sentiment[prediction][k] += [sentiment_score[k]]
    # task 5
    # What are the characteristics of rumour-creating users, and are they different to normal users?
    
    pm = tweet.includes["users"][0].public_metrics
    # public_metrics[prediction] = {public_metrics[prediction][k] + [pm[k]] for k in pm.keys()}
    for k in pm.keys():
        public_metrics[prediction][k] += [pm[k]]
    
    # extra 1 at analysis
    # What are the characteristics of rumour-creating users, and are they different to normal users?
    
    tweet_ats = at_analysis(text)
    for a in tweet_ats:
        ats[prediction][a] = ats[prediction].get(a, 0) + 1


def main():
    vectorizer = TfidfVectorizer()
    train_file = "train.data.txt"
    train_file = pd.read_csv('./%s.csv'%train_file,keep_default_na=False)
    train_data = train_file['main_tweet']
    train_data.replace('', np.nan, inplace=True)
    train_data.dropna(inplace=True)
    train_data = train_data.apply(lambda x: tokenize_tweet(x))
    vectorizer.fit(train_data.tolist())
    # print(train_data)
    model = keras.models.load_model("tfidf.h5")
    file = "covid.data.txt"
    file_open = open('./project-data/' + file, 'r')
    counter = 0
    for line in file_open.readlines():
        line = line.strip()
        ids = line.split(',')
        source_id = ids[0]
        # id = 1243807211815649285
        get_tweet(source_id, model, vectorizer)
        print(counter)
        counter += 1
        # if counter > 3:
        #     break
    
    # printing out result
    for i in range(0, len(topic_time)):
        print(f"Topic from {time_step[i]} to {time_step[i+1]} are \n {topic_time[i]}\n")
    
    metrics = [topics, hashtags, sentiment, public_metrics, count, ats]
    metric_names = ["topics", "hashtags", "sentiment", "public_metrics", "count", "ats"]
    for k in range(0,len(metrics)):
        for p in [0,1]:
            clf = "rumour" if p else "nonrumour"
            print(f"The {metric_names[k]} for {clf} are \n {metrics[k][p]}")
            print()
    
    tasks = [topics, hashtags, ats]
    file_name = ["topics", "hashtags", "ats"]
    label = ["topic", "hashtag", "user"]
    num = 0
    for task in tasks:
        task_index = list(set(list(task[0].keys()) + list(task[1].keys())))
        task_dict = {"nonrumour":[], "rumour": []}
        for i in task_index:
            task_dict["nonrumour"] = task_dict["nonrumour"] + [task[0].get(i, 0)]
            task_dict["rumour"] = task_dict["rumour"] + [task[1].get(i, 0)]
        task_df = pd.DataFrame(data=task_dict, index = task_index)
        print(task_df)
        task_df.to_csv(f"output/{file_name[num]}.csv",index=True, index_label=label[num])
        num += 1
        
    trend_index = []
    for d in topic_time:
        trend_index = trend_index + list(d.keys())
    trend_index = list(set(trend_index))
    trend_dict = {"2020/1/1-2020/6/1":[], "2020/6/1-2021/1/1": [], "2021/1/1-2021/6/1": [], 
                   "2021/6/1-2022/1/1": [], "2022/1/1-2022/6/1": []}
    for i in trend_index:
        foo = 0
        for k in trend_dict.keys():
            trend_dict[k] = trend_dict[k] + [topic_time[foo].get(i, 0)]
            foo += 1
    trend_df = pd.DataFrame(data=trend_dict, index = trend_index)
    print(trend_df)
    trend_df.to_csv("output/trend.csv",index=True, index_label="Topic")
    
        
    sentiment_nonrumour = pd.DataFrame(data=sentiment[0])
    sentiment_rumour = pd.DataFrame(data=sentiment[1])
    print(sentiment_nonrumour)
    print(sentiment_rumour)
    sentiment_nonrumour.to_csv("output/sentiment_nonrumour.csv",index=False)
    sentiment_rumour.to_csv("output/sentiment_rumour.csv",index=False)

    public_metrics_nonrumour = pd.DataFrame(data=public_metrics[0])
    public_metrics_rumour = pd.DataFrame(data=public_metrics[1])
    print(public_metrics_nonrumour)
    print(public_metrics_rumour)
    public_metrics_nonrumour.to_csv("output/public_metrics_nonrumour.csv",index=False)
    public_metrics_rumour.to_csv("output/public_metrics_rumour.csv",index=False)
    
    count_dict = {"nonrumour": count[0], "rumour": count[1],
                  "2020/1/1-2020/6/1":count_time[0], 
                  "2020/6/1-2021/1/1": count_time[1], 
                  "2021/1/1-2021/6/1": count_time[2], 
                  "2021/6/1-2022/1/1": count_time[3], 
                  "2022/1/1-2022/6/1": count_time[4]}
    count_df = pd.DataFrame(data = count_dict, index=[0])
    count_df.to_csv("output/count.csv", index=True, index_label="0")
    
main()