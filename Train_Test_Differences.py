import pandas as pd

remove_atts = ['Unnamed: 0', 'sessionId', 'visitId', 'visitStartTime', 'fullVisitorId', 'id_real', 'networkDomain']
train_df = pd.read_csv('Data/reduced_data.csv', low_memory=False).drop(remove_atts, axis=1)
test_df = pd.read_csv('Data/reduced_data_test.csv', low_memory=False).drop(remove_atts, axis=1)


train_df = train_df.select_dtypes(include='object')
test_df = test_df.select_dtypes(include='object')

train_df._combine_match_columns

train_df.columns

from Experimental_Setup import  FunctionFeaturizer

ff = FunctionFeaturizer()

maps_train = ff.fit(train_df,train_df['transactionRevenue'])


fet = maps_train.featurizers
type(fet)
len(fet)
test_df['channelGrouping'].value_counts()

for i in range(len(fet)):
    att = fet[i].columns[0]
    m = fet[i]
    t = test_df[att].value_counts()
    print(m)
    print("SE UPOREDJUJE SA")
    print("  ")
    print(t)
    print("  ")

# channelGrouping, medium
fet[0]
test_df['channelGrouping'].value_counts()


fet[11]
test_df['medium'].value_counts()

