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
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
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
    'classify__n_estimators':[100, 200],
    'classify__max_depth':[5,10]
}


clf = GridSearchCV(pipe, param_grid = param_grid, cv=4, scoring = 'roc_auc', n_jobs=4, return_train_score=False)
model=clf.fit(df.drop(['fullVisitorId', 'transactionRevenue'], axis=1), y_classification)
pickle._dump(model, open('Models/First_Logistic.sav', 'wb'))

#TODO Store predictions from classification

'''
Regression model fitting
'''




class GroupEstimator(BaseEstimator):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self._base_estimator = clone(self.base_estimator)
        y_true = y.drop('fullVisitorId', axis=1)
        self._base_estimator.fit(X, y_true)
        return self

    def predict(self, X):
        predictions = self._base_estimator.predict(X)
        #predictions = predictions.clip(min=0)
        predictions = np.abs(predictions)
        return predictions


def performance_on_user(y_true, y_pred, measure=mean_squared_error):
    y_true['y_pred'] = y_pred
    aggregated = y_true.groupby('fullVisitorId').agg('sum').reset_index()
    print(aggregated.head())
    true = np.log1p(aggregated['transactionRevenue'])
    pred = np.log1p(aggregated['y_pred'])
    rmse = np.sqrt(measure(true, pred))
    return (rmse)

rmse_score = make_scorer(performance_on_user)


mask_positive_cases = (y_regression>0)
y = y_regression[mask_positive_cases]
X = df[mask_positive_cases]



pipe = Pipeline(steps = [('reducer', ReduceCardinality()), ('featurize', FunctionFeaturizer()) , ('impute',SimpleImputer(strategy='median')), ('classify', GroupEstimator(XGBRegressor()))])

'''
param_grid = {
    'classify__n_estimators':[100, 200, 300, 500],
    'classify__max_depth':[3, 5, 7, 10]
}

'''

'''
param_grid_XGboost ={
    'classify_eta':[0.01, 0.1, 0.3, 0.5, 0.7],
    'classify_gama':[0.01, 0.05, 0.1, 0.3, 0.5, 1, 5],
    'classify_max_depth':[3,5,10,15,20],
    'classify_colsample_bytree':[0.1, 0.5, 0.9],
    'classify_colsample_bylevel':[0.1, 0.5, 0.9],
    'classify_lambda': [0.1, 0.3, 0.5, 0.9]
    'classify_alpha': [0.1, 0.3, 0.5, 0.9]
}

'''
'''
base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=nan, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1). Check the list of available parameters with `estimator.get_params().keys()`.
'''



param_grid_XGboost ={
    'classify__base_estimator__gamma':[0, 0.3],
    'classify__base_estimator__max_depth':[5],
    'classify__base_estimator__n_estimators':[20],
    'classify__base_estimator__nthread':[4]
}


clf = GridSearchCV(pipe, cv=4,param_grid = param_grid_XGboost, scoring=rmse_score, return_train_score=False, error_score='raise', verbose=True)
model=clf.fit(X=X.drop(['fullVisitorId', 'transactionRevenue'], axis =1), y=X[['fullVisitorId', 'transactionRevenue']])

model.best_estimator_.named_steps['classify']
print(model.best_score_)


pickle._dump(model, open('Models/First_RF_Regression.sav', 'wb'))


'''
Learning on whole dataframe
'''
clf = GridSearchCV(pipe, cv=4,param_grid = param_grid_XGboost, scoring=rmse_score, return_train_score=False, error_score='raise', verbose=True)
model=clf.fit(X=df.drop(['fullVisitorId', 'transactionRevenue'], axis =1), y=df[['fullVisitorId', 'transactionRevenue']])
print(model.best_score_)
pickle._dump(model, open('Models/Total_Regression.sav', 'wb'))






