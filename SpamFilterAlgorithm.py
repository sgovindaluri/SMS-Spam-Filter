#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: sonalig

"""

import nltk
import numpy as np

from sklearn import preprocessing

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction import text

from sklearn import metrics 
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
 



###Read data into dataframe###

df = pd.read_table('data/SMSSpamCollection', header=None)
origLabels = df[0]
rawText = df[1]
le = preprocessing.LabelEncoder()
encodedLabels = le.fit_transform(origLabels)


###Preprocessing###

processedText = rawText.str.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b',
                                  'emailaddr')
processedText = processedText.str.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)',
                                   'httpaddr')
processedText = processedText.str.replace(r'£|\$', 'moneysymb')    
processedText = processedText.str.replace(
    r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
    'phonenumbr')    
processedText = processedText.str.replace(r'\d+(\.\d+)?', 'numbr')
processedText = processedText.str.replace(r'[^\w\d\s]', ' ')
processedText = processedText.str.replace(r'\s+', ' ')
processedText = processedText.str.replace(r'^\s+|\s+?$', '')
processedText = processedText.str.lower()

###Stop word removal##

stop_words = nltk.corpus.stopwords.words('english')
processedText = processedText.apply(lambda x: ' '.join(
    term for term in x.split() if term not in set(stop_words)))

### STEMMING ###

porter = nltk.PorterStemmer()
processedText = processedText.apply(lambda x: ' '.join(
        porter.stem(term) for term in x.split()))


## Feature Extraction : TF-IDF Vectorizer
vectorizer = text.TfidfVectorizer(ngram_range=(1,1))
X_ngrams = vectorizer.fit_transform(processedText)


X_train, X_test, y_train, y_test = train_test_split(X_ngrams,
                                                    encodedLabels,
                                                    test_size=0.2,
                                                    random_state = 42,
                                                    stratify=encodedLabels)

### CLASSIFICATION ###
#1. Support Vector Machine#
clf = svm.LinearSVC(loss='hinge')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score_svm = metrics.f1_score(y_test, y_pred)


print('\n\n')
print('SVM Preliminary Analysis: Confusion Matrix')
print('------------------------------------------')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred),
             index=[['actual', 'actual'], ['spam', 'ham']],
             columns=[['predicted', 'predicted'], ['spam', 'ham']]))
print('\nSVM Preliminary Analysis: F1 Score')
print('-----------------------------------')
print(score_svm)

#2. Multinomial Naive Bayes#
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score_nb = metrics.f1_score(y_test, y_pred)

print('\n\n')
print('NB Preliminary Analysis: Confusion Matrix')
print('-----------------------------------------')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred),
             index=[['actual', 'actual'], ['spam', 'ham']],
             columns=[['predicted', 'predicted'], ['spam', 'ham']]))
print('\nNB Preliminary Analysis: F1 Score')
print('----------------------------------')
print(score_nb)
print('\n')




## K-fold Validation 


param_grid = [{'C': np.logspace(-4, 4, 20)}]

grid_search = GridSearchCV(estimator=svm.LinearSVC(loss='hinge'),
                           param_grid=param_grid,
                           cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
                           scoring='f1',
                           n_jobs=-1)

scores = cross_val_score(estimator=grid_search,
                         X=X_ngrams,
                         y=encodedLabels,
                         cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
                         scoring='f1',
                         n_jobs=-1)


print('\n\n')
print('Validation Scores')
print('-----------------')
print(scores)
print('Mean Score', scores.mean())
print('\n\n')
#####


grid_search.fit(X_train, y_train)
valid_clf = svm.LinearSVC(loss='hinge', C=grid_search.best_params_['C'])
valid_clf.fit(X_train, y_train)
y_pred = valid_clf.predict(X_test)
test_error = metrics.f1_score(y_test, y_pred)
print('\n\n')
print('F1 score on Testing Set', test_error)
 

# Print Confusion Matrix
print('\n\n')
print('SVM Confusion Matrix')
print('--------------------')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred),
             index=[['actual', 'actual'], ['spam', 'ham']],
             columns=[['predicted', 'predicted'], ['spam', 'ham']]))
print('\n\n')



# print 10 relevant features

grid_search.fit(X_ngrams, encodedLabels)
final_clf = svm.LinearSVC(loss='hinge', C=grid_search.best_params_['C'])
final_clf.fit(X_ngrams, encodedLabels);

print('\n\n')
print('Top Ten relevant features')
print('-------------------------')
print(pd.Series(
    final_clf.coef_.T.ravel(),
    index=vectorizer.get_feature_names()).sort_values(ascending=False)[:10])


