
# Import Libraries
import dask.dataframe as dd
import pandas as pd

# Import csv files and create set of company names
media = dd.read_csv('C:\\Users\ChrisGomes\Projects\MediaStock\media0*.csv')
companies = pd.read_csv('C:\\Users\ChrisGomes\Projects\MediaStock\companyNamesFormatted.csv')
companyNames = set(companies['Name']) # convert company names to a set

# Dummy Codes and Derived Features
media['MentionsPerSource'] = media.NumMentions / media.NumSources # derived attribute of mentions per source
media['InUS'] = media.CountryCode.map(lambda x: 1 if x == 'US' else 0) # dummy code for if occured in US
media['NASDAQActor1'] = media.Actor1Name.map(lambda x: 1 if x in companyNames else 0) # dummy code if actor 1 in NASDAQ
media['NASDAQActor2'] = media.Actor2Name.map(lambda x: 1 if x in companyNames else 0) # dummy code if actor 2 in NASDAQ

# Min-Max Normalization


# Export all media to new csv
media.to_csv(path_or_buf='C:\\Users\ChrisGomes\Projects\MediaStock\media.csv')
