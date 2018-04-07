# %% Import Libraries
import pandas as pd
import numpy as np
import datetime as dt
%matplotlib inline

# %% Import Data Sets
nasdaq = pd.read_csv('C:\\Users\ChrisGomes\Projects\MediaStock\\nasdaqFull.csv') # Source: https://finance.yahoo.com/quote/%5EIXIC/history?ltr=1
media = pd.read_csv('C:\\Users\ChrisGomes\Projects\MediaStock\media.csv') # Source: GDELT Project on Google CLoud Platform

# %% Stock Price Date Formating and Aggregation
nasdaq['Date'] = pd.to_datetime(nasdaq['Date']) # convert to date type in data frame
nasdaq['MonthYear'] = nasdaq.Date.dt.to_period('M') # extract month and year and format it to match media data set
nasdaq['MonthYear'] = nasdaq['MonthYear'].astype('str').str.replace('-', '').astype(int)
nasdaqMonthly = nasdaq.groupby(by='MonthYear')['Close'].mean().reset_index()

# %% Data Plots and Outliers for Media Features
media.plot.scatter(x='MonthYear', y='GoldsteinScale') # Plot for Goldstein Scale feature of media data set
media.plot.scatter(x='MonthYear', y='NumMentions') # Plot for NumMentions feature of media data set
media.plot.scatter(x='MonthYear', y='NumSources') # Plot for NumSources feature of media data set
media.plot.scatter(x='MonthYear', y='AvgTone') # Plot for AvgTone feature of media data set
