# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:42:56 2019

@author: Shriyash Shende
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

s= pd.read_csv('sms_raw_NB.csv',encoding = "ISO-8859-1")
s.info()
s['type'].describe()

# cleaning data 
import re
stop_words = []
with open("Stop Words.txt") as f:
    stop_words = f.read()

stop_words = stop_words.split("\n")

def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))


s.text = s.text.apply(cleaning_text)


def split_into_words(i):
    return [word for word in i.split(" ")]

from sklearn.model_selection import train_test_split

s_train,s_test = train_test_split(s,test_size=0.3)
s_bow = CountVectorizer(analyzer=split_into_words).fit(s.text)

all_emails_matrix = s_bow.transform(s.text)
all_emails_matrix.shape # (5559,6661)
# For training messages
s_emails_matrix = s_bow.transform(s_train.text)
s_emails_matrix.shape # (3891,6661)



# For testing messages
test_emails_matrix = s_bow.transform(s_test.text)
test_emails_matrix.shape


####### Without TFIDF matrices ########################
# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(s_emails_matrix ,s_train.type)
train_pred_m = classifier_mb.predict(s_emails_matrix )
accuracy_train_m = np.mean(train_pred_m==s_train.type) 

test_pred_m = classifier_mb.predict(test_emails_matrix)
accuracy_test_m = np.mean(test_pred_m==s_test.type) 

# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(s_emails_matrix.toarray(),s_train.type.values) 
train_pred_g = classifier_gb.predict(s_emails_matrix.toarray())
accuracy_train_g = np.mean(train_pred_g==s_train.type) 

test_pred_g = classifier_gb.predict(test_emails_matrix.toarray())
accuracy_test_g = np.mean(test_pred_g==s_test.type) 

###################### With TFIDF matrices############
# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(s_emails_matrix )

train_tfidf.shape

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_emails_matrix)

test_tfidf.shape #  

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf,s_train.type)
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m==s_train.type) 

test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m==s_test.type) 

# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_tfidf.toarray(),s_train.type.values) 
train_pred_g = classifier_gb.predict(train_tfidf.toarray())
accuracy_train_g = np.mean(train_pred_g==s_train.type)
test_pred_g = classifier_gb.predict(test_tfidf.toarray())
accuracy_test_g = np.mean(test_pred_g==s_test.type)
