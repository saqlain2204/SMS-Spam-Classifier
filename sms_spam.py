# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

df = pd.read_csv("SMSSpamCollection", sep ='\t', names = ['label', 'message'])
df

df.isna().any()

df['label'] = df['label'].apply(lambda x: 0 if x=='ham' else 1)

df

import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer

import re
from nltk.corpus import stopwords
nltk.download('wordnet')
corpus = []
wl = WordNetLemmatizer()
for i in range(0, len(df)):
  review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
  review = review.lower()
  review = review.split()

  review = [wl.lemmatize(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  corpus.append(review)

corpus

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2500)
X = cv.fit_transform(corpus).toarray()

X

df['label'].value_counts()

y = df['label']

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42, k_neighbors = 5)
X_res, y_res = sm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

y_res.value_counts()

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)

confusion_m

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

accuracy

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

