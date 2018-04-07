
# Import Libraries
import dask.dataframe as dd
import pandas as pd

# Import csv files
media = dd.read_csv('C:\\Users\ChrisGomes\Projects\MediaStock\media0*.csv')
companyNames = pd.read_csv('C:\\Users\ChrisGomes\Projects\MediaStock\companyNamesFormatted.csv')

# %% Dummy Codes and Derived Features
media['MentionsPerSource'] = media.NumMentions / media.NumSources # derived attribute of mentions per source
media['InUS'] = media.CountryCode.map(lambda x: 1 if x == 'US' else 0) # dummy code for if occured in US

def actorsInNasdaq(df):
    if (df['Actor1Name'] in companyNames['Name'] or df['Actor2Name'] in companyNames['Name']):
        return 1
    else:
        return 0

media[]


# Export all media to new csv
media.to_csv(path_or_buf='C:\\Users\ChrisGomes\Projects\MediaStock\media.csv')
