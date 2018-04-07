
# Import Libraries
import pandas as pd

# Import first file
defaultPath = 'C:\\Users\ChrisGomes\Projects\MediaStock\media00000000000'
media = pd.read_csv(defaultPath + str(0) + '.csv')

# Import and append the rest of the files
for i in range(1, 7):
    media = media.append(pd.read_csv(defaultPath + str(i) + '.csv'))

# Export all media to new csv
media.to_csv(path_or_buf='C:\\Users\ChrisGomes\Projects\MediaStock\media.csv')
