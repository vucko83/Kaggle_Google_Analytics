import pandas as pd
from datetime import datetime as dt
pd.set_option('display.max_columns', 100)
import numpy as np



def from_dicts_to_df(dicts):
    '''
    :param dicts: Column of dicts 
    :return: Pandas Dataframe
    '''
    geos_eval = list(dicts.apply(lambda s: eval(s)))
    df = pd.DataFrame.from_records(geos_eval)
    return (df)

def replaceTF(str):
    '''
    Replacing true, false with True, False
    :param str: 
    :return: 
    '''
    str_r = str.replace('false', 'False')
    str_r = str_r.replace('true', 'True')
    return(str_r)

def date_features(date):
    '''
    :param date: date integer 
    :param dt: datetime object
    :return: 
    '''
    # TODO extract weekends, holidays

    date = dt.strptime(str(date), '%Y%m%d')
    day = date.day
    weekday = date.weekday()
    month = date.month
    year = date.year
    weeknum = date.isocalendar()[1]

    date_dict = {'year':year, 'month':month, 'weekday':weekday, 'weeknum':weeknum, 'day':day}
    return (str(date_dict))


def flatten_data(df):

    # Check how to keep ids for merge

    totals_df= from_dicts_to_df(df['totals'])
    geo_df = from_dicts_to_df(df['geoNetwork'])
    false_replaced = df['device'].apply(replaceTF)

    device_df = from_dicts_to_df(false_replaced)
    dates_featurized = df['date'].apply(date_features)
    date_df = from_dicts_to_df(dates_featurized)

    traffic_replaced = df['trafficSource'].apply(replaceTF)
    trafficSource_df = from_dicts_to_df(traffic_replaced)

    adwordsClickInfo_df = from_dicts_to_df(trafficSource_df['adwordsClickInfo'].apply(str))
    trafficSource_df = trafficSource_df.drop('adwordsClickInfo', axis=1)

    #data=pd.concat([totals_df, geo_df, device_df, date_df, trafficSource_df, adwordsClickInfo_df], sort=False, axis = 1)
    data = pd.concat([totals_df, geo_df, device_df, date_df, trafficSource_df, adwordsClickInfo_df], axis=1)


    return(data)


'''
Drop columns
'''
# Define function for counting number of appearance for every categorical value
def num_of_occurrence_in_cat(data):

    logical_vec = (data.dtypes == object)
    result_dict = {}   # pd.DataFrame(columns=['att_name','category','count'])

    for i in range(len(logical_vec)):

        if logical_vec[i] == True:
            att = data.columns[i]
            count_val = pd.DataFrame(pd.value_counts(data.iloc[:, i]))
            result_dict.update({att : count_val})

    return result_dict

# we can return list of attributes also
def remove_single_category(data):
    cat_counts = num_of_occurrence_in_cat(data)
    one_value_att = []
    for key in cat_counts.keys():
        if len(cat_counts[key]) == 1:
            one_value_att.append(cat_counts[key].columns[0])

    data = data.drop(columns=one_value_att)
    return (data, one_value_att)

def set_types(data):
    data['visitNumber'] = pd.to_numeric(data['visitNumber'], errors = 'coerce')
    data['hits'] = pd.to_numeric(data['hits'], errors = 'coerce')
    data['bounces'] = pd.to_numeric(data['bounces'], errors = 'coerce')
    data['newVisits'] = pd.to_numeric(data['newVisits'], errors = 'coerce')
    data['visits'] = pd.to_numeric(data['visits'], errors = 'coerce')
    data['pageviews'] = pd.to_numeric(data['pageviews'], errors = 'coerce')
    #data['transactionRevenue'] = pd.to_numeric(data['transactionRevenue'], errors = 'coerce')
    data['visitStartTime'] = data['visitStartTime'].apply(dt.fromtimestamp)
    return(data)

def map_to_other_categories(df, count_dict, threshold):
    atts = df.columns
    for key in count_dict.keys():
        if ((key in atts) and (key not in ['fullVisitorId', 'sessionId',  'gclId'])):
            percentage = count_dict[key]/np.sum(count_dict[key])*100
            low_percentage_values = percentage.index[percentage[key] < threshold].tolist()
            df[key] = df[key].apply(lambda x: 'other' if x in low_percentage_values else x)
    return (df)

def fill_nas(df):
    df['isVideoAd'] = df['isVideoAd'].fillna(True)
    df['isTrueDirect'] = df['isTrueDirect'].fillna(False)
    #df['transactionRevenue'] = df['transactionRevenue'].fillna(0)
    return df



df = pd.read_csv('Data/train.csv', low_memory=False,  dtype={'fullVisitorId': 'object'})



flattened_data = flatten_data(df)
flattened_data.columns
flattened_data.head()

flat_data = df.drop(['totals', 'geoNetwork', 'device', 'date', 'trafficSource'], axis=1)


#all_data = pd.concat([flat_data, flattened_data], sort = False, axis=1)
all_data = pd.concat([flat_data, flattened_data], axis=1)


all_data = set_types(all_data)
all_data = fill_nas(all_data)
all_data = all_data.drop('targetingCriteria', axis =1)

all_data.columns

all_data, remove_atts = remove_single_category(all_data)

all_data.shape

category_counts = num_of_occurrence_in_cat(all_data)

#category_counts.keys()


# TODO have to create map (!!!! Wrapper)
all_data = map_to_other_categories(all_data, category_counts, 5)

#######
null_analysis = (all_data.isnull().sum()/all_data.shape[0]) * 100

all_data['bounces'] = all_data['bounces'].fillna(0)
all_data['newVisits'] = all_data['newVisits'].fillna(0)
all_data['pageviews'] = all_data['pageviews'].fillna(0)

all_data.columns
to_remove = ['adContent', 'keyword', 'adNetworkType', 'gclId', 'page', 'slot', 'referralPath']

all_data = all_data.drop(to_remove, axis=1)


all_data['id_real'] = all_data['fullVisitorId']+all_data['visitStartTime'].astype(str)

null_analysis = (all_data.isnull().sum()/all_data.shape[0]) * 100

all_data.to_csv('Data/reduced_data.csv', header = True)

########################### Apply same procedure on test ###################

df = pd.read_csv('Data/test.csv', low_memory=False,  dtype={'fullVisitorId': 'object'})

flattened_data = flatten_data(df)

flat_data = df.drop(['totals', 'geoNetwork', 'device', 'date', 'trafficSource'], axis=1)

#all_data = pd.concat([flat_data, flattened_data], sort = False, axis=1)
all_data = pd.concat([flat_data, flattened_data], axis=1)

all_data = set_types(all_data)
all_data = fill_nas(all_data)
all_data = all_data.drop('targetingCriteria', axis =1)


all_data, remove_atts = remove_single_category(all_data)
category_counts = num_of_occurrence_in_cat(all_data)

#category_counts.keys()


# TODO have to create map (!!!! Wrapper)
all_data = map_to_other_categories(all_data, category_counts, 5)


#######
null_analysis = (all_data.isnull().sum()/all_data.shape[0]) * 100

all_data['bounces'] = all_data['bounces'].fillna(0)
all_data['newVisits'] = all_data['newVisits'].fillna(0)
all_data['pageviews'] = all_data['pageviews'].fillna(0)


to_remove = ['adContent', 'keyword', 'adNetworkType', 'gclId', 'page', 'slot', 'referralPath']

all_data = all_data.drop(to_remove, axis=1)
all_data['id_real'] = all_data['fullVisitorId']+all_data['visitStartTime'].astype(str)

null_analysis = (all_data.isnull().sum()/all_data.shape[0]) * 100

all_data.to_csv('Data/reduced_data_test.csv', header = True)