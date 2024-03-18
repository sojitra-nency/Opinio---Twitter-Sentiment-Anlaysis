import streamlit as st
import pickle
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

port_stem = PorterStemmer()

def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]',' ',content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

with open("trained_model.sav", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pickle", "rb") as f:
    vectorizer = pickle.load(f)

st.title("Sentiment Analysis App")

tweet = st.text_input("Enter a tweet for prediction:")

if tweet:
    processed_tweet = stemming(tweet)  
    tweet_vector = vectorizer.transform([processed_tweet])
    prediction = model.predict(tweet_vector)

    if prediction[0] == 0:
        st.write("Sentiment: Negative")
    else:
        st.write("Sentiment: Positive")
