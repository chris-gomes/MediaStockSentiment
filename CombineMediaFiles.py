
# Import Libraries
import pandas as pd

# Import and combine files
defaultPath = 'C:\\Users\ChrisGomes\Projects\MediaStock\media00000000000'
media = pd.read_csv(defaultPath + str(0) + '.csv')

for i in range(1, 7):
    media = media.append(pd.read_csv(defaultPath + str(i) + '.csv'))

print(media.count)

# Export all media to new csv
media.to_csv(path_or_buf='C:\\Users\ChrisGomes\Projects\MediaStock\media.csv')
