# %% Import Libraries
import pandas as pd
import numpy as np
import datetime as dt
%matplotlib inline

# %% Import Data Sets
nasdaq = pd.read_csv('.\data\\nasdaqFull.csv') # Source: https://finance.yahoo.com/quote/%5EIXIC/history?ltr=1
media = pd.read_csv('.\data\mediaFormatted.csv') # Source: GDELT Project on Google CLoud Platform

# %% Normalize the given column (Min-Max)
def minMaxNormalize(column):
    """ Function to find a normalized form of the given column.
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
nasdaqMonthly = nasdaq.groupby(by='MonthYear')['Close'].mean().reset_index()

media.describe()

# %% Data Plots and Outliers for Media Features
media.plot.scatter(x='MonthYear', y='GoldsteinScale') # Plot for Goldstein Scale feature of media data set
media.plot.scatter(x='MonthYear', y='MentionsPerSource') # Plot for NumSources feature of media data set
media.plot.scatter(x='MonthYear', y='AvgTone') # Plot for AvgTone feature of media data set
media.plot.scatter(x='MonthYear', y='InUS') # Plot for InUS feature of media data set
media.plot.scatter(x='MonthYear', y='NASDAQActor1') # Plot for NASDAQActor1 feature of media data set
media.plot.scatter(x='MonthYear', y='NASDAQActor2') # Plot for NASDAQActor2 feature of media data set

# %% Min-Max Normalization for Features
media['GoldsteinScale'] =
