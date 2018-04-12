
# Import Libraries
import pandas as pd
import numpy as np


def main():
	#Import csv files and create set of company names
	media = pd.read_csv('./data/media.csv',
			    dtype={'Actor1Name':str,
				'Actor2Name':str,
				'GoldsteinScale':np.float64,
				'NumMentions':np.float64,
				'NumMentions':np.float64,
				'NumSources':np.float64,
				'AvgTone':np.float64,
				'CountryCode':str}) # Source: GDELT on Google Cloud Platform
	companies = pd.read_csv('./data/companyNamesFormatted.csv') # Source: https://www.nasdaq.com/screening/company-list.aspx
	companyNames = set(companies.Name) # convert company names to a set
	print('Import done') # status output

	# Dummy Codes and Derived Features (if any of the data is null it is recorded as not in the US or NASDAQ)
	media['MentionsPerSource'] = media.NumMentions / media.NumSources # derived attribute of mentions per source
	media['InUS'] = np.where(media.CountryCode=='US', 1, 0) # dummy code for if occured in US
	media['NASDAQActor1'] = media['NASDAQActor1'].apply(lambda x: 1 if str(x) in companyNames else 0) # dummy code if actor 1 in NASDAQ
	media['NASDAQActor2'] = media['NASDAQActor2'].apply(lambda x: 1 if str(x) in companyNames else 0) # dummy code if actor 2 in NASDAQ
	print('Codes done') # status output

	# remove rows with NAs (decided since the data set has a significant number of rows and imputation would not provide much benefit)
	media = media.dropna(axis = 0, how = 'any')
	columns = ['MonthYear', 'GoldsteinScale', 'AvgTone', 'MentionsPerSource', 'InUS', 'NASDAQActor1', 'NASDAQActor2']
	for name in columns: # check if rows with NAs were actually removed
		if media[name].isna().any():
			print('NAs in ' + name)
	print('NA removal done') # status output

	# convert MonthYear to int
	media['MonthYear'] = media.MonthYear.astype('int64')
	print('MonthYear conversion done')

	# Aggreate media data frame to average out GoldsteinScale, AvgTone, and MentionsPerSource and
	# add up occurances InUS, NASDAQActor1 and NASDAQActor2
	media = media.groupby('MonthYear', as_index=False).agg(
		{'GoldsteinScale':'mean','AvgTone':'mean','MentionsPerSource':'mean',
		'InUS':'sum', 'NASDAQActor1':'sum', 'NASDAQActor2':'sum'})
	print(media.dtypes)
	print('Agg done') # status check

	# Export all media to new csv
	media.to_csv(path_or_buf='./data/mediaFormatted.csv')

if __name__ == '__main__':
	main()
