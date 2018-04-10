
# Import Libraries
import pandas as pd

# Import company names csv
companyList = pd.read_csv('.\data\companyList.csv')

# Remove duplicate names (there are duplicates because companies can have multiple tickers)
names = pd.DataFrame(pd.Series.unique(companyList['Name']), columns=['Name'])

# Make all uppercase
names['Name'] = names['Name'].str.upper()

# Remove punctuation
names['Name'] = names['Name'].str.replace(".", "")
names['Name'] = names['Name'].str.replace(",", "")

# Add Google to list to account for name change to Alphabet
google = pd.DataFrame([['GOOGLE'],['INVESTOR']], columns=['Name'])
names = names.append(google)

# Export names out to a new csv
names.to_csv(path_or_buf='.\data\companyNamesFormatted.csv')
