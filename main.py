import numpy as np
import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

column_names = ["target", "id", "date", "flag", "user", "text"]
twitter_data = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", names=column_names)

twitter_data.replace({'target':{4:1}}, inplace=True)

port_stem = PorterStemmer()

def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]',' ',content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

print("======================================")
twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)
print("======================================")

X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("======================================")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
print("======================================")

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("Accuracy: ", training_data_accuracy)

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Accuracy: ", testing_data_accuracy)

with open('vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)
    
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open("trained_model.sav","rb"))

X_new = X_test[132435]
print(Y_test[132435])

prediction = model.predict(X_new)
print(prediction)

if prediction[0] == 0: print("Negative")
else: print("Positive")
