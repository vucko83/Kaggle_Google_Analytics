import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from Data_Cleansing import Cleansing_Tools as ct
from Feature_Engineering.Categorical_Quantification import ReduceCardinality
from Feature_Engineering.Categorical_Quantification import FunctionFeaturizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor



from Feature_Engineering.Categorical_Quantification import DataFrameImputer
import pickle

#TODO Set threshold that captures most of the positive cases, Optimize everything

pd.set_option('display.max_columns', 100)
df = pd.read_csv('Data/reduced_data.csv', low_memory=False, dtype={'fullVisitorId':'object'})


'''
Extract visitorId and transactionRevenue 
'''

y_regression= df['transactionRevenue']
y_classification = y_regression.apply(lambda x: 1 if x>0 else x)

'''
Classification model fitting
'''

#X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, stratify=y)


pipe = Pipeline(steps = [('reducer', ReduceCardinality()), ('featurize', FunctionFeaturizer()) , ('impute',SimpleImputer(strategy='median')), ('classify', RandomForestClassifier())])

param_grid = {
    'classify__n_estimators':[100, 200, 300, 500],
    'classify__max_depth':[3, 5, 7, 10]
}


clf = GridSearchCV(pipe, param_grid = param_grid, cv=4, scoring = 'roc_auc', n_jobs=4, return_train_score=False)
model=clf.fit(df.drop(['fullVisitorId', 'transactionRevenue'], axis=1), y_classification)
pickle._dump(model, open('Models/First_Logistic.sav', 'wb'))


'''
Regression model fitting
'''

mask_positive_cases = (y_regression>0)
y = y_regression[mask_positive_cases]
X = df[mask_positive_cases]



pipe = Pipeline(steps = [('reducer', ReduceCardinality()), ('featurize', FunctionFeaturizer()) , ('impute',SimpleImputer(strategy='median')), ('classify', RandomForestRegressor())])

param_grid = {
    'classify__n_estimators':[100, 200, 300, 500],
    'classify__max_depth':[3, 5, 7, 10]
}


param_grid_XGboost ={
    'classify_eta':[0.01, 0.1, 0.3, 0.5, 0.7]

}


clf = GridSearchCV(pipe, param_grid = param_grid, cv=4, scoring ='neg_mean_squared_error', n_jobs=4, return_train_score=False)

model=clf.fit(X=X.drop(['fullVisitorId', 'transactionRevenue'], axis=1), y=np.log(y))

pickle._dump(model, open('Models/First_RF_Regression.sav', 'wb'))


