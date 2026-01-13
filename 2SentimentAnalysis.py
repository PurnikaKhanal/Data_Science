from textblob import TextBlob
wiki = TextBlob("Python is a great programming language.")
print(wiki.tags) #For part-of-speech tagging
print(wiki.words) #For word tokenization
print(wiki.sentiment.polarity) #For sentiment analysis

# This is a sample code for Twitter Sentiment Analysis using TextBlob library in Python.
# It analyzes the sentiment of tweets containing a specific keyword.
