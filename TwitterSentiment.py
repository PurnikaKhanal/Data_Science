from textblob import TextBlob
wiki = TextBlob("Python is a great programming language.")
print(wiki.tags) #For part-of-speech tagging
print(wiki.words) #For word tokenization
print(wiki.sentiment.polarity) #For sentiment analysis