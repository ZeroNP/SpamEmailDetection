# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:33:02 2020

@author: Shreyas
"""

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

dataset = pd.read_csv('emails.csv')
dataset.drop_duplicates(inplace=True)

stop_words = [
"a", "about", "above", "across", "after", "afterwards", 
"again", "all", "almost", "alone", "along", "already", "also",    
"although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "as", "at", "be", "became", "because", "become","becomes", "becoming", "been", "before", "behind", "being", "beside", "besides", "between", "beyond", "both", "but", "by","can", "cannot", "cant", "could", "couldnt", "de", "describe", "do", "done", "each", "eg", "either", "else", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "find","for","found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next","no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part","perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should","since", "sincere","so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take","than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they",
"this", "those", "though", "through", "throughout",
"thru", "thus", "to", "together", "too", "toward", "towards",
"under", "until", "up", "upon", "us",
"very", "was", "we", "well", "were", "what", "whatever", "when",
"whence", "whenever", "where", "whereafter", "whereas", "whereby",
"wherein", "whereupon", "wherever", "whether", "which", "while", 
"who", "whoever", "whom", "whose", "why", "will", "with",
"within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
]


def process_text(text):
    '''Remove punctuatuon and stopwords (useless words)'''
    text_without_punctuation = [char for char in text if char not in string.punctuation]
    joined_text = ''.join(text_without_punctuation)
    
    clean_words = [word for word in joined_text.split() 
                   if word.lower() not in stop_words] 
    
    return clean_words

dataset['text'].head().apply(process_text)

from sklearn.feature_extraction.text import CountVectorizer
messages_bow = CountVectorizer(analyzer=process_text).fit_transform(dataset['text'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(messages_bow, 
                                                    dataset['spam'],
                                                    test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

pred = classifier.predict(X_train)
print("Classification Report: \n", classification_report(y_train, pred))

print('Confusion Matrix: \n', confusion_matrix(y_train, pred))

print('Accuracy Score: \n', accuracy_score(y_train, pred))

pred_test = classifier.predict(X_test)
print("Classification Report: \n", classification_report(y_test, pred_test))

print('Confusion Matrix: \n', confusion_matrix(y_test, pred_test))

print('Accuracy Score: \n', accuracy_score(y_test, pred_test))

