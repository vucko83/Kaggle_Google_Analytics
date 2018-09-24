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

    data=pd.concat([totals_df, geo_df, device_df, date_df, trafficSource_df, adwordsClickInfo_df], sort=False, axis = 1)



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
    data['transactionRevenue'] = pd.to_numeric(data['transactionRevenue'], errors = 'coerce')
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
    df['transactionRevenue'] = df['transactionRevenue'].fillna(0)
    return df



df = pd.read_csv('Data/train.csv', low_memory=False,  dtype={'fullVisitorId': 'object'})

flattened_data = flatten_data(df)

flattened_data.head()

flat_data = df.drop(['totals', 'geoNetwork', 'device', 'date', 'trafficSource'], axis=1)


all_data = pd.concat([flat_data, flattened_data], sort = False, axis=1)


all_data = set_types(all_data)

all_data = fill_nas(all_data)


all_data = all_data.drop('targetingCriteria', axis =1)

all_data, remove_atts = remove_single_category(all_data)

category_counts = num_of_occurrence_in_cat(all_data)

#category_counts.keys()

all_data=map_to_other_categories(all_data, category_counts, 5)

#######
null_analysis = (all_data.isnull().sum()/all_data.shape[0]) * 100

all_data['bounces'] = all_data['bounces'].fillna(0)
all_data['newVisits'] = all_data['newVisits'].fillna(0)
all_data['pageviews'] = all_data['pageviews'].fillna(0)

to_remove = ['adContent', 'keyword', 'adNetworkType', 'gclId', 'page', 'slot', 'referralPath']

all_data = all_data.drop(to_remove, axis=1)

all_data = all_data.drop('referralPath', axis=1)

all_data['id_real'] = all_data['fullVisitorId']+all_data['visitStartTime'].astype(str)

null_analysis = (all_data.isnull().sum()/all_data.shape[0]) * 100

all_data.to_csv('reduced_data.csv', header = True)

all_data = pd.read_csv('reduced_data.csv')

all_data.head()

all_data['referralPath']

a=

all_data.dtypes

len(a.unique())

len(all_data['visitId'].unique())

len(all_data['sessionId'].unique())

all_data.shape

all_data.dtypes

pd.value_counts(all_data['referralPath'])






all_data.head(100)

# Id resiti
# Izbaciti one sa samo 1 kategorijom (mala varijansa i kod kategorickih i numerickih)
# longitude lattitude
#

all_data.dtypes
all_data.shape # (6,325,571, 60)

all_data.columns
all_data.describe()

'''
Check null values in data 
'''

# Vratiti se na ovo i razmotriti kako da se resi problem ovoliko nedostajucih vrednosti
num_rows = all_data.shape[0]
100 - (all_data.count()/num_rows * 100)

null_analysis = (all_data.isnull().sum()/num_rows) * 100

all_data.dtypes





attributes_to_remove = ''

d['visitStartTime']

d.dtypes
a.keys()

dt.date(d['visitStartTime'])

data = remove_single_category(all_data)



d.dtypes

null_analysis = (d.isnull().sum()/num_rows) * 100

all_data = all_data.drop(['targetingCriteria'],axis=1)

a=num_of_occurrence_in_cat(all_data)

a['channelGrouping']

all_data.dtypes

all_data['hits']


### Problem with targetingCriteria attributes (error: TypeError: unhashable type: 'dict')
### Check non null vales of this attributes
all_data['targetingCriteria'][all_data['targetingCriteria'].notnull()]

### All values are {} (empty Dictionaries)
### remove this attribute from data
all_data = all_data.drop(['targetingCriteria'],axis=1)


cat_counts = num_of_occurrence_in_cat(all_data)
cat_counts.keys()



for key in cat_counts.keys():
    print(key +": " + str(len(cat_counts[key])))


### DA LI TREBA ISKLJUCITI I OVA DVA ATRIBUTA
k = ['isTrueDirect','isVideoAd']

for key in k:
    print(cat_counts[key])

# remove attributes that have only one value for all non null rows



null_analysis = (all_data.isnull().sum()/num_rows) * 100
cat_counts


all_data.dtypes


all_data.visitId.count()
all_data.sessionId.count()
all_data.transactionRevenue.count()


len(all_data.visitId.unique())
len(all_data.sessionId.unique())


'''
# provera sa pocetnim podacima

df.dtypes
df.shape # (903,653, 12)

df.columns
df.describe()

org_cat_counts = num_of_occurrence_in_cat(df)
org_cat_counts.keys()

for key in org_cat_counts.keys():
    print(key +": " + str(len(org_cat_counts[key])))

cat_counts['socialEngagementType']
'''
