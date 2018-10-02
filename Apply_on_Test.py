import pandas as pd
import pickle
import numpy as np
from Data_Cleansing import Cleansing_Tools as ct
from Feature_Engineering.Categorical_Quantification import ReduceCardinality
from Feature_Engineering.Categorical_Quantification import FunctionFeaturizer
from Feature_Engineering.Categorical_Quantification import DataFrameImputer
from sklearn.impute import SimpleImputer

########################### Apply same procedure on test ###################

train_df= pd.read_csv('Data/reduced_data.csv', low_memory=False,  dtype={'fullVisitorId': 'object'}, nrows=10)



test = pd.read_csv('Data/reduced_data_test.csv', low_memory=False,  dtype={'fullVisitorId': 'object'})

'''
Classification
'''
LR = pickle.load(open('Models/First_Logistic.sav','rb'))
predictions_class = LR.predict(test.drop('fullVisitorId', axis=1)) # Predict 0s with classification algorithm


test['Predicted_Revenue'] = predictions_class

# Wrap up 0 predictions
predictions_0 = test[test['Predicted_Revenue']==0][['fullVisitorId', 'Predicted_Revenue']]


data_for_regression = test[test['Predicted_Revenue']==1].drop('Predicted_Revenue', axis=1)

'''
Regression
'''

#visitor_ids_positive = predictions_1['fullVisitorId']
#len(visitor_ids_positive.unique())

RF1 = pickle.load(open('Models/First_RF_Regression.sav','rb'))

prediction_regression = RF1.predict(data_for_regression.drop('fullVisitorId', axis=1))

data_for_regression['Predicted_Revenue'] = np.exp(prediction_regression)

predictions_1 = data_for_regression[['fullVisitorId', 'Predicted_Revenue']]




att_names = ['fullVisitorId', 'PredictedLogRevenue']

all_predictions = pd.concat([predictions_0, predictions_1], axis=0)
all_predictions.columns = att_names


for_submission = all_predictions.groupby('fullVisitorId')['PredictedLogRevenue'].sum().reset_index()

final_predictions =np.log1p(for_submission['PredictedLogRevenue'])

for_submission['PredictedLogRevenue']=final_predictions


for_submission.to_csv('Data/Submissions/first_submission.csv', header = True, index=False)

'''
Error analyses
'''

plt.scatter(range(len(prediction_regression)), np.sort(prediction_regression))
plt.scatter(range(for_submission.shape[0]), np.sort(for_submission['PredictedLogRevenue']))
plt.show()
plt.close()





