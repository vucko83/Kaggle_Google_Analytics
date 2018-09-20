import pandas as pd
from datetime import datetime as dt

df = pd.read_csv('Data/train.csv', nrows = 10000)


'''
Geo information to Data Frame
'''
type(df['geoNetwork'][0])


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
    date_dict = {'year':year, 'month':month, 'weekday':weekday, 'day':day}
    return (str(date_dict))


totals_df= from_dicts_to_df(df['totals'])
geo_df = from_dicts_to_df(df['geoNetwork'])
false_replaced = df['device'].apply(replaceTF)

device_df = from_dicts_to_df(false_replaced)
dates_featurized = df['date'].apply(date_features)
date_df = from_dicts_to_df(dates_featurized)

traffic_replaced = df['trafficSource'].apply(replaceTF)
trafficSource_df = from_dicts_to_df(traffic_replaced)

adwordsClickInfo_df = from_dicts_to_df(trafficSource_df['adwordsClickInfo'].apply(str))

str(trafficSource_df['adwordsClickInfo'])

a=pd.concat([totals_df, geo_df])

totals_df.shape
geo_df.shape
a.shape

trafficSource_df['adwordsClickInfo']

trafficSource_df

# TODO Dates to day, week, month, quarter, semi-year, year

for att in df.columns:
    print(att+ ': ' + str(df[att][9999]))

df['trafficSource'][9999]

df.shape

df.columns

type(a)



ex_date = df['date'][9970]
ex_date

df
a=dt.strptime(str(ex_date), '%Y%m%d')
a.


a.weekday()
df.columns
a.month
a.day
a.year



df.columns



#false_replaced = df['device'].apply(lambda str: str.replace("false", "False"))

false_replaced[1]



geo_df.iloc[2, :]



df.columns
type(geos_eval)



