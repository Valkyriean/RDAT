from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
text = '''
Not anymore and hopefully not again #lockdown #covid #corona #fitzroy #graffiti #slogangraffiti @graffiterati @sevenbreaths
'''

res = sia.polarity_scores(text)

print(res)