
# Import Libraries
import pandas as pd

# Import company names csv
companyList = pd.read_csv('C:\\Users\ChrisGomes\Projects\MediaStock\companyList.csv')

# Keep only name column
names = companyList[['Name']]

# Remove duplicate names (there are duplicates because companies can have multiple tickers)
names['Name'] = pd.Series.unique(names['Name'])

# Make all uppercase
names['Name'] = pd.Series.str.upper(names['Name'])

# Remove punctuation
names['Name'] = pd.Series.str.replace(".", "")
names['Name'] = pd.Series.str.replace(",", "")

# Add Google to list to account for name change to Alphabet
google = pd.DataFrame([['GOOGLE']], columns='Name')
names = names.append(google)

# Export names out to a new csv
names.to_csv(path_or_buf='C:\\Users\ChrisGomes\Projects\MediaStock\companyNamesFormatted.csv')
