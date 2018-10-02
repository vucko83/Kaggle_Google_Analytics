import pandas as pd
from Data_Cleansing import Cleansing_Tools as ct
pd.set_option('display.max_columns', 100)

'''
Prepare Train
'''

def train_basic_preparation(df, path):
    flattened_data = ct.flatten_data(df)
    flat_data = df.drop(['totals', 'geoNetwork', 'device', 'date', 'trafficSource'], axis=1) #Remove duplicated attributes

    all_data = pd.concat([flat_data, flattened_data], axis=1)
    all_data = ct.set_types(all_data, True)

    all_data = all_data.drop('targetingCriteria', axis =1) # TODO Check this empty dictionaries
    all_data = ct.fill_na_bool(all_data) # Fills two features where only True or False are present

    all_data['transactionRevenue'] = all_data['transactionRevenue'].fillna(0)

    to_remove = ['adContent', 'keyword', 'adNetworkType', 'gclId', 'page', 'slot', 'referralPath', 'city', 'sessionId', 'visitId', 'visitStartTime', 'networkDomain', 'continent']
    # continent wasn't find in test set check this probably becaus of remove_single_category - revise method for train and test
    all_data = all_data.drop(to_remove, axis=1)
    all_data, remove_atts = ct.remove_single_category(all_data)
    atts = list(all_data.drop('transactionRevenue', axis=1).columns)
    all_data.to_csv(path, header = True, index=False)

    return (atts)



def test_basic_preparation(df, path, atts):
    flattened_data = ct.flatten_data(df)
    flat_data = df.drop(['totals', 'geoNetwork', 'device', 'date', 'trafficSource'], axis=1) #Remove duplicated attributes

    all_data = pd.concat([flat_data, flattened_data], axis=1)
    all_data = ct.set_types(all_data, train=False)

    all_data = all_data.drop('targetingCriteria', axis =1) # TODO Check this empty dictionaries
    all_data = ct.fill_na_bool(all_data) # Fills two features where only True or False are present

    #to_remove = ['adContent', 'keyword', 'adNetworkType', 'gclId', 'page', 'slot', 'referralPath', 'city', 'sessionId', 'visitId', 'visitStartTime', 'networkDomain', 'continent']
    # continent wasn't find in test set check this probably becaus of remove_single_category - revise method for train and test
    all_data = all_data[atts]

    all_data.to_csv(path, header = True, index=False)


train_path = 'Data/reduced_data.csv'
test_path =  'Data/reduced_data_test.csv'

df = pd.read_csv('Data/train.csv', low_memory=False,  dtype={'fullVisitorId': 'object'})

atts = train_basic_preparation(df, train_path)

df = pd.read_csv('Data/test.csv', low_memory=False,  dtype={'fullVisitorId': 'object'})
test_basic_preparation(df, test_path, atts)








