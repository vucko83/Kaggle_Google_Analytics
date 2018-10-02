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


def set_types(data, train = True):
    data['visitNumber'] = pd.to_numeric(data['visitNumber'], errors = 'coerce')
    data['hits'] = pd.to_numeric(data['hits'], errors = 'coerce')
    data['bounces'] = pd.to_numeric(data['bounces'], errors = 'coerce')
    data['newVisits'] = pd.to_numeric(data['newVisits'], errors = 'coerce')
    data['visits'] = pd.to_numeric(data['visits'], errors = 'coerce')
    data['pageviews'] = pd.to_numeric(data['pageviews'], errors = 'coerce')
    data['visitStartTime'] = data['visitStartTime'].apply(dt.fromtimestamp)
    if train:
        data['transactionRevenue'] = pd.to_numeric(data['transactionRevenue'], errors='coerce')
    return(data)


def fill_na_bool(df):
    df['isVideoAd'] = df['isVideoAd'].fillna(True)
    df['isTrueDirect'] = df['isTrueDirect'].fillna(False)
    #df['transactionRevenue'] = df['transactionRevenue'].fillna(0)
    return df
