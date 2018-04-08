
# %% Import Libraries
import dask.dataframe as dd
import pandas as pd

# %% Import csv files
media = dd.read_csv('C:\\Users\ChrisGomes\Projects\MediaStock\media0*.csv')
companyNames = pd.read_csv('C:\\Users\ChrisGomes\Projects\MediaStock\companyNamesFormatted.csv')

# %% Dummy Codes and Derived Features
media['MentionsPerSource'] = media.NumMentions / media.NumSources # derived attribute of mentions per source
media['InUS'] = media.CountryCode.apply(lambda x: 1 if x == 'US' else 0) # dummy code for if occured in US
media['NASDAQActor1'] = media.Actor1Name.apply(lambda x: 1 if companyNames['Name'].str.contains(str(x)).any() else 0) # dummy code if actor 1 in NASDAQ
media['NASDAQActor2'] = media.Actor2Name.apply(lambda x: 1 if companyNames['Name'].str.contains(str(x)).any() else 0) # dummy code if actor 2 in NASDAQ

media.head(5)

# Export all media to new csv
media.to_csv(path_or_buf='C:\\Users\ChrisGomes\Projects\MediaStock\media.csv')
