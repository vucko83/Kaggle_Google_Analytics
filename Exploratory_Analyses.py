import pandas as pd

df = pd.read_csv('Data/train.csv', nrows = 10000)
df.columns
df.dtypes
example = df['geoNetwork'][2]

example

# Types of data


# Percentage of Missings per Attribute


#

'''
Visualization of outcome
'''
import numpy as np
from matplotlib import pyplot as plt

train = pd.read_csv('Data/reduced_data.csv', low_memory=False,  dtype={'fullVisitorId': 'object'})

grouped_outcome = train.groupby('fullVisitorId')['transactionRevenue'].sum().reset_index()

outcome = grouped_outcome[grouped_outcome['transactionRevenue']>0]

outcome = np.log(outcome['transactionRevenue'])
outcome = outcome['transactionRevenue']


plt.scatter(range(len(outcome)) , np.sort(outcome))
plt.show()
plt.close()

for_submission = pd.read_csv('submission.csv')

for_submission.head()
pred_outcome = for_submission[(for_submission['PredictedLogRevenue']<100 ) & ( for_submission['PredictedLogRevenue']>0 )]
#pred_outcome = for_submission[for_submission['PredictedLogRevenue']>1000 ]

plt.scatter(range(pred_outcome.shape[0]), np.sort(pred_outcome['PredictedLogRevenue']))
plt.show()