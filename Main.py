# %% Import Libraries
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
%matplotlib inline

# %% Import Data Sets
nasdaq = pd.read_csv('.\data\\nasdaqFull.csv') # Source: https://finance.yahoo.com/quote/%5EIXIC/history?ltr=1
media = pd.read_csv('.\data\mediaFormatted.csv') # Source: GDELT Project on Google CLoud Platform

# %% Normalize a given column (Min-Max)
def minMaxNormalize(column):
    """ Function to create a normalized form of the given column.
    Args:
        column (numpy series): column of numeric values
    Returns:
        The column with the normalized values in the same order.
    """
    min = column.min()
    max = column.max()
    return (column - min) / (max - min)

# %% Stock Price Date Formating and Aggregation
nasdaq['Date'] = pd.to_datetime(nasdaq['Date']) # convert to date type in data frame
nasdaq['MonthYear'] = nasdaq.Date.dt.to_period('M') # extract month and year and format it to match media data set
nasdaq['MonthYear'] = nasdaq['MonthYear'].astype('str').str.replace('-', '').astype(int)
nasdaqMonthly = nasdaq.groupby(by='MonthYear')['Close'].mean().reset_index() # group closing price by month

# Finds if closing prices increased for models that try to predict if prices will decrease or increase in the next month
nasdaqMonthly['NextMonthClose'] = nasdaqMonthly.Close.shift(periods=-1) # add next month's close to previous line in new column
nasdaqMonthly['Increased'] = nasdaqMonthly.NextMonthClose > nasdaqMonthly.Close # see if price increased
nasdaqMonthly['Increased'] = nasdaqMonthly.Increased.apply(lambda x: 1 if x else 0) # change to a 1 0 encoding
nasdaqMonthly = nasdaqMonthly.dropna(axis = 0, how = 'any')

# %% Merging Media and Stock Prices and Sort Result
all = pd.merge(left=media, right=nasdaqMonthly, on='MonthYear', sort=True)

# %% Data Plots and Outliers for Media Features
all.plot.scatter(x='MonthYear', y='GoldsteinScale') # Plot for Goldstein Scale feature of media data set
all.plot.scatter(x='MonthYear', y='MentionsPerSource') # Plot for NumSources feature of media data set
all.plot.scatter(x='MonthYear', y='AvgTone') # Plot for AvgTone feature of media data set
all.plot.scatter(x='MonthYear', y='InUS') # Plot for InUS feature of media data set
all.plot.scatter(x='MonthYear', y='NASDAQActor1') # Plot for NASDAQActor1 feature of media data set
all.plot.scatter(x='MonthYear', y='NASDAQActor2') # Plot for NASDAQActor2 feature of media data set

# %% Remove Outliers
all = all[all.MentionsPerSource < 6.0] # removed outlier in MentionsPerSource

# %% Min-Max Normalization for Features
columnNames = media.columns[2:len(media.columns)] # get column names to normalize
for name in list(columnNames): # normalize each of the columns
    media[name] = minMaxNormalize(media[name])

# %% Train and Validation Subsetting
both = model_selection.train_test_split(all, train_size=0.7, random_state=11, shuffle=True) # get array with both training and validation
trainAttributes = both[0][['GoldsteinScale', 'MentionsPerSource', 'AvgTone', 'InUS', 'NASDAQActor1', 'NASDAQActor2']]
trainResults = both[0]['Increased']
validationAttributes = both[1][['GoldsteinScale', 'MentionsPerSource', 'AvgTone', 'InUS', 'NASDAQActor1', 'NASDAQActor2']]
validationResults = both[1]['Increased']

# %% kNN Prediction of Future Stock Price Movement
results = [] # create array to hold resulting scores
for k in [1:20]: # try kNN algorithm for multiple Ks
    kNNModel = KNeighborsClassifier(n_neighbors=k, n_jobs=4) # create model
    kNNModel.fit(X=trainAttributes, y=trainResults) # train the model
    score = kNNModel.score(X=validationAttributes, y=validationResults) # test against validation
    result[k] = score
