#!/usr/bin/env python3

# %% Import Libraries
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
%matplotlib inline

# %% Import Data Sets
nasdaq = pd.read_csv('.\data\\nasdaqFull.csv') # Source: https://finance.yahoo.com/quote/%5EIXIC/history?ltr=1
media = pd.read_csv('.\data\mediaFormatted.csv') # Source: GDELT Project on Google CLoud Platform

# %% Functions Used In Analysis and Predicitons
def minMaxNormalize(column):
    """ Function to create a normalized version of the given column.
    Args:
        column (series): column of numeric values
    Returns (series):
        The column with the normalized values in the same order.
    """
    min = column.min()
    max = column.max()
    return (column - min) / (max - min)

def testModel(name, tA, tR, vA, k=None, kernel=None):
    """ Function to run and test a model.
    Args:
        name (string): name of the model to use (kNN, Logistic Regression, or SVM)
        tA (data frame/series): attributes of the training data
        tR (series): correct results for training data
        vA (data frame/series): attributes of the validation data
        k (int): optional number of closestneighbors for kNN models
        kernel (string): optional name of kernel for SVM models
    Returns (series):
        Series of the predicitions made by the model for the validation data.
    """
    if name == 'kNN': # creates kNN model
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=4)
    elif name == 'Logistic Regression': # creates Logistic Regression model
        model = LogisticRegression(random_state=8)
    elif name == 'SVM': # creates SVM model
        model = SVC(kernel=kernel, random_state=8)
    elif name == 'KMeans': # create KMeans model
        model = KMeans(n_clusters=2, random_state=8)
    model.fit(X=tA, y=tR) # train model
    return model.predict(X=vA) # return series of predicitons

def percentEqual(predictions, actuals):
    """ Function to check the percent of occurs that are equal between the
        predictions and actual values.
    Args:
        predicitions (series): Prediction values from a models
        actuals (series): Actual values that occured for the predictions.
    Returns (float):
        Percent of correct predictions equal to actual values.
    """
    equal = predictions == actuals
    counts = equal.apply(lambda x: 1 if x else 0)
    return float(counts.sum() / counts.size)

# %% Stock Price Date Formating and Aggregation
nasdaq['Date'] = pd.to_datetime(nasdaq['Date']) # convert to date type in data frame
nasdaq['MonthYear'] = nasdaq.Date.dt.to_period('M') # extract month and year and format it to match media data set
nasdaq['MonthYear'] = nasdaq['MonthYear'].astype('str').str.replace('-', '').astype(int)
nasdaqMonthly = nasdaq.groupby(by='MonthYear')['Close'].mean().reset_index() # group closing price by month

# Finds if closing prices increased or decreased in the next month
nasdaqMonthly['NextMonthClose'] = nasdaqMonthly.Close.shift(periods=-1) # add next month's close to previous line in new column
nasdaqMonthly['Increased'] = nasdaqMonthly.NextMonthClose > nasdaqMonthly.Close # see if price increased
nasdaqMonthly['Increased'] = nasdaqMonthly.Increased.apply(lambda x: 1 if x else 0) # change to a 1 0 encoding
nasdaqMonthly = nasdaqMonthly.dropna(axis = 0, how = 'any')

# %% Merging Media and Stock Prices and Sort Result
all = pd.merge(left=media, right=nasdaqMonthly, on='MonthYear', sort=True)

# %% Data Plots and Outliers for Media Features
featureNames = all.columns[2:8] # column names of all features
for name in list(featureNames):
    all.plot.scatter(x='MonthYear', y=name) # Plot the given feature

# %% Remove Outliers
all = all[all.MentionsPerSource < 6.0] # removed outlier in MentionsPerSource
all.plot.scatter(x='MonthYear', y='MentionsPerSource') # check for shape after removal

# %% Min-Max Normalization for Features
for name in list(featureNames): # normalize each of the columns
    all[name] = minMaxNormalize(all[name])

# %% Train and Validation Subsetting
# Only GoldsteinScale, AvgTone and MentionsPerSource were kept as attributes since
# the others had significant increasing trends after 2005, which could throw off the model predicitons
both = model_selection.train_test_split(all, train_size=0.7, random_state=11, shuffle=True) # get array with both training and validation
trainAttributes = both[0][['GoldsteinScale', 'AvgTone', 'MentionsPerSource']]
trainResults = both[0]['Increased']
validationAttributes = both[1][['GoldsteinScale', 'AvgTone', 'MentionsPerSource']]
validationResults = both[1]['Increased']

# %% kNN Prediction of Future Stock Price Movement
scores = [] # create array to hold resulting scores
for k in range(1,21): # try kNN algorithm for multiple Ks
    kNNPred = testModel(name='kNN',
                        tA=trainAttributes,
                        tR=trainResults,
                        vA=validationAttributes,
                        k=k) # run model and get predictions
    scores.append(percentEqual(kNNPred, validationResults))
resultskNN = pd.DataFrame({'k':list(range(1,21)), 'PercentCorrect':scores})
resultskNN.plot.scatter(x='k', y='PercentCorrect') # plot percent correct over k
kNNPred = testModel(name='kNN',
                    tA=trainAttributes,
                    tR=trainResults,
                    vA=validationAttributes,
                    k=3) # run model and get predictions
resultskNN.loc[resultskNN.PercentCorrect.idxmax(),] # highest performing k value

# %% Logistic Regression Prediction of Future Stock Price Movement
logRegPred = testModel(name='Logistic Regression',
    tA=trainAttributes,
    tR=trainResults,
    vA=validationAttributes) # run model and get predictions
percentEqual(logRegPred, validationResults) # get percent correct

# %% SVM Prediction of Future Stock Price Movement
kernels = ['linear', 'rbf', 'poly']
scores = []
for kernel in kernels:
    svmPred = testModel(name='SVM',
                        tA=trainAttributes,
                        tR=trainResults,
                        vA=validationAttributes,
                        kernel=kernel) # run model and get predictions
    scores.append(percentEqual(svmPred, validationResults)) # get percent correct
svmResults = pd.DataFrame({'kernels':list(range(1,4)), 'PercentCorrect':scores})
svmResults.plot.scatter(x='kernels', y='PercentCorrect') # plot percent correct for each kernal
svmResults.loc[svmResults.PercentCorrect.idxmax(),] # highedt performing kernal value (all were equal)

# %% Stacked Ensemble model (have each of the models vote on whether prices will increase)
votesForIncrease = kNNPred + logRegPred + svmPred # combine all votes for an increase
votesForIncrease = pd.Series(data=votesForIncrease, dtype=np.int64) # convert to series for easy calculations
finalDecisions = votesForIncrease.apply(lambda x: 1 if x >= 2 else 0) # find all cases where majority voted for an increase
validationResults = pd.Series(validationResults.tolist()) # reset indices on validation results
percentEqual(finalDecisions, validationResults) # find percent of correct guesses
confusion_matrix(y_true=validationResults, y_pred=finalDecisions) # shows that prediciton was always yes

# %% Final Analysis in Model Comparison and Prediciton Results
"""
In comparing the models, the difference in prediction percentages were minimal.
In the end, the biggest observation when comparing the models is the fact that the
Logistic Regression,SVM, and the stacked ensembled models all had the same percentage
correctness. This occured because both the logistic regression and SVM models predicted
that the closed price would always increase, which could have occured due to an
underlining trend in the data and could hint towards the idea in finance that in the
long-term stock prices always rise. In the end, it does not appear that these
attributes are useful in prediciting where closing prices will be during the next month.
The kNN model, for example, was only correct about 62% of the time, which is only
marginally better than blindly guessing. Going forward, this process could be
attempted with other attributes that may be able to increase the prediction ability.
"""
