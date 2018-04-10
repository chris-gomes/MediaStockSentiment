# Import Libraries
import pandas as pd
import numpy as np

# Import csv files and create set of company names
defaultPath = '.\data\media0000000000{0:02d}.csv' # base path for all media csvs
media = pd.read_csv(defaultPath.format(0),
    dtype={'MonthYear': np.int64,
    'GoldsteinScale': np.float64,
    'NumMentions': np.float64,
    'NumSources': np.float64,
    'AvgTone': np.float64}) # add first media csv

for i in range(1, 16):
    media = media.append(pd.read_csv(defaultPath.format(i),
                                    dtype={'MonthYear': np.int64,
                                    'GoldsteinScale': np.float64,
                                    'NumMentions': np.float64,
                                    'NumSources': np.float64,
                                    'AvgTone': np.float64}),
                        ignore_index=True) # append each csv numbered after 0

companies = pd.read_csv('.\data\companyNamesFormatted.csv') # Source: https://www.nasdaq.com/screening/company-list.aspx
companyNames = set(companies.Name) # convert company names to a set

# Dummy Codes and Derived Features
media['MentionsPerSource'] = media.NumMentions / media.NumSources # derived attribute of mentions per source
media['InUS'] = np.where(media.CountryCode=='US', 1, 0) # dummy code for if occured in US
media['NASDAQActor1'] = np.where(media.Actor1Name in companyNames, 1, 0) # dummy code if actor 1 in NASDAQ
media['NASDAQActor2'] = np.where(media.Actor2Name in companyNames, 1, 0) # dummy code if actor 2 in NASDAQ

# check if NAs are present in any of the attribute columns
attributeCols = ['GoldsteinScale', 'AvgTone', 'MentionsPerSource', 'InUS', 'NASDAQActor1', 'NASDAQActor2']
for name in attributeCols:
    if pandas.isnull(media[name]).any():
        print('NAs present in ' + name)

# Aggreate media data frame to average out GoldsteinScale, AvgTone, and MentionsPerSource and
# add up occurances InUS, NASDAQActor1 and NASDAQActor2
goldToneMSGrouped = media.groupby('MonthYear')['GoldsteinScale', 'AvgTone', 'MentionsPerSource'].mean() # aggregate GoldsteinScale, AvgTone, and MentionsPerSource
inUSNasdaqGrouped = media.groupby('MonthYear')['InUS', 'NASDAQActor1', 'NASDAQActor2'].sum() # aggregate InUS, NASDAQActor1 and NASDAQActor2
mediaGrouped = pd.merge(left = goldToneMSGrouped, right = inUSNasdaqGrouped, left_on='MonthYear', right_on='MonthYear') # merge columns back together

# Export all media to new csv
mediaGrouped.to_csv(path_or_buf='.\data\media.csv')
