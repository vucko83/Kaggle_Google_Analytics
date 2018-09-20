import pandas as pd

df = pd.read_csv('Data/train.csv', nrows = 10000)
df.columns
df.dtypes
example = df['geoNetwork'][2]

example

# Types of data


# Percentage of Missings per Attribute


#