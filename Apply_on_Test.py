import pandas as pd
import pickle


test = pd.read_csv('Data/reduced_data_test.csv', low_memory=False)

test.columns
# channelGrouping, medium

visitor_id = test['fullVisitorId']
remove_atts = ['Unnamed: 0', 'sessionId', 'visitId', 'visitStartTime', 'fullVisitorId', 'id_real', 'networkDomain']
test = test.drop(remove_atts, axis=1)
test = test.fillna(0)

test['channelGrouping'] = test['channelGrouping'].replace('Display', 'other')
test['medium'] = test['medium'].replace('cpc', 'other')

test['channelGrouping'].value_counts()
test['medium'].value_counts()
# Take Ids


'''
Classification
'''
LR = pickle.load(open('Models/First_Logistic.sav','rb'))

# Predict 0s with classification algorithm

predictions_0 = LR.predict(test)

# Concatenate predictions with Ids
type(predictions_0)

test['pred'] = predictions_0.tolist()
test['fullVisitorId'] = visitor_id

group_test = pd.DataFrame(test.groupby('fullVisitorId')['pred'].sum())

mask = group_test.pred >= 1
column_name = 'pred'
group_test.loc[mask, column_name] = 1

# Select Ids with 0s
# Create variable with TRUE if nationality is USA
test_0 = group_test[group_test['pred'] == 0]
test_1 = group_test[group_test['pred'] == 1]


test_1 = test_1.reset_index()

result = pd.merge(test_1, test, on='fullVisitorId',how='inner')
result = result.drop(['pred_x','pred_y'], axis=1)

result.to_csv('Data/purchased_predictions.csv', header = True)

'''
Regression
'''
RF = pickle.load(open('Models/First_RF_Regression.sav','rb'))




# Assign 0s to predicted 0s



# Predict values with regression




