import tweepy
from textblob import TextBlob

# Use bearer token for authentication
bearer_token = "Your_bearer_token_here"

client = tweepy.Client(bearer_token=bearer_token)

# Search recent tweets (up to 10) containing the keyword "Python"
tweets = client.search_recent_tweets(
    query="Python -is:retweet lang:en",
    max_results=10
)

# Analyze sentiment for each tweet
for tweet in tweets.data:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print("Polarity:", analysis.sentiment.polarity)
    print("-" * 40)
