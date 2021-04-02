# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:50:59 2020
https://pythonprogramming.net/features-labels-machine-learning-tutorial/?completed=/regression-introduction-machine-learning-tutorial/
@author: Eoin
"""

import pandas as pd
import numpy as np
import quandl
import math
import pickle
from sklearn import preprocessing, svm #Support Vector Regression
from sklearn import model_selection as cross_validation# has changed since tutorial
from sklearn.linear_model import LinearRegression
df = quandl.get('WIKI/GOOGL')
# df = pd.read_csv('wiki_google.csv')
print(df.head())
#%%
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
#%%
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())
#%%
#features are descriptive attributes - label is what you're trying to predict/forecast
forecast_col = 'Adj. Close'
df.fillna(value=-99_999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df))) #1% the length of the dataset

df['label'] = df[forecast_col].shift(-forecast_out) # shift the label column up 
len(df['label'])
#https://pythonprogramming.net/training-testing-machine-learning-tutorial/?completed=/features-labels-machine-learning-tutorial/
#%%

X = np.array(df.drop(['label'], 1)) # everything except label

X = preprocessing.scale(X)# Generally, you want your features in machine learning to be in a range of -1 to 1
X_lately = X[-forecast_out:]# so the predicted data has the same scaling
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])
#%%

#%%since this also shuffles your data for you?? why. I'm gonna add shuffle=False. Turns out that's bad, negative confidence
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                                                     y,
                                                                     test_size=0.2)#,
                                                                      # shuffle=False)# train, then test
#%%
# clf = svm.SVR()#classifier
# clf.fit(X_train, y_train)
# confidence = clf.score(X_test, y_test)
# print(confidence)
#%%
# clf = LinearRegression(n_jobs=-1)#use all available threads
# clf.fit(X_train, y_train)
# confidence = clf.score(X_test, y_test)
# print(confidence)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)
#%%
# for k in ['linear','poly','rbf','sigmoid']:
#     clf = svm.SVR(kernel=k)
#     clf.fit(X_train, y_train)
#     confidence = clf.score(X_test, y_test)
#     print(k,confidence)
#%%
forecast_set = clf.predict(X_lately)
#%%
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
df['Forecast'] = np.nan
#%%
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
#%%
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
#%%
plt.figure()
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
#%%

# with open('linearregression.pickle','wb') as f:
#     pickle.dump(clf, f)
    