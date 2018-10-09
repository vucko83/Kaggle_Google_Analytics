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
from lightgbm import LGBMRegressor

#TODO Set threshold that captures most of the positive cases, Optimize everything

pd.set_option('display.max_columns', 100)
df = pd.read_csv('Data/reduced_data.csv', low_memory=False, dtype={'fullVisitorId':'object'}, nrows = 100000)



#TODO Store predictions from classification

'''
Regression model fitting
'''


class GroupEstimator(BaseEstimator):
    def __init__(self, base_estimator, threshold=0.5, name='positive_probability'):
        self.base_estimator = base_estimator
        self.threshold=threshold
        self.positive_mask = None
        self.negative_mask = None
        self.name = name


    def fit(self, X, y):

        self._base_estimator = clone(self.base_estimator)
        self.positive_mask = np.array(X['positive_probability']>=self.threshold)
        y_true = y[self.positive_mask]
        X_true = X[self.positive_mask]
        print(type(y_true))
        print(type(X_true))
        self._base_estimator = self._base_estimator.fit(X_true.drop('positive_probability', axis=1), y_true.drop(['fullVisitorId'], axis=1))
        print('Finish Fit')

        return self

    def predict(self, X):
        print('ENTERING PREDICT')
        print(type(X))

        self.positive_mask = np.array(X['positive_probability'] >= self.threshold)
        X_positives = X[self.positive_mask]
        print('X_positives selected')
        predictions_1 = self._base_estimator.predict(X_positives.drop('positive_probability', axis=1))

        #predictions_1 = self._base_estimator.predict(X_positives)
        X_positives['predictions'] =predictions_1

        for_join = pd.DataFrame(X_positives['predictions'])

        print('before_join')
        X= X.join(for_join, how='left')
        X['predictions']= X['predictions'].fillna(0)

        #predictions = predictions.clip(min=0)
        predictions = np.abs(X['predictions'])
        return predictions


def performance_on_user(y_true, y_pred, measure=mean_squared_error):
    print('Entering Performance on user')
    y_true['y_pred'] = y_pred
    aggregated = y_true.groupby('fullVisitorId').agg('sum').reset_index()
    true = np.log1p(aggregated['transactionRevenue'])
    pred = np.log1p(aggregated['y_pred'])
    rmse = np.sqrt(measure(true, pred))
    return (rmse)

rmse_score = make_scorer(performance_on_user, greater_is_better=True)


'''
Light GBM regression

'''

param_grid_LGBM = {
    'classify__base_estimator__objective':['regression'],
    'classify__base_estimator__boosting_type':['gbdt'],
    'classify__base_estimator__metric':['rmse'],
    'classify__base_estimator__n_estimators':[200, 500, 1000],#, 300, 500, 1000, 5000], #10000
    'classify__base_estimator__num_leaves':[50, 70, 100],
    'classify__base_estimator__learning_rate':[00.1, 0.1], #0.01
    'classify__base_estimator__bagging_fraction':[0.7, 1],#0.8
    'classify__base_estimator__feature_fraction':[0.7, 1],#.3
    'classify__base_estimator__max_depth':[5, 10, 15, 20] #-1
}


from Feature_Engineering.Categorical_Quantification import PositiveEstimator
from tempfile import mkdtemp
from shutil import rmtree

LR = pickle.load(open('Models/First_Logistic.sav','rb'))
RF = LR.best_estimator_.named_steps['classify']
cachedir = mkdtemp()


X=df.drop(['fullVisitorId', 'transactionRevenue'], axis =1)
y=df[['fullVisitorId', 'transactionRevenue']]


pipe = Pipeline(steps = [('reducer', ReduceCardinality()), ('featurize', FunctionFeaturizer()), ('impute', DataFrameImputer()), ('positive', PositiveEstimator(RF)), ('classify', GroupEstimator(LGBMRegressor(n_jobs=4)))], memory=cachedir)
clf = GridSearchCV(pipe, cv=5,param_grid = param_grid_LGBM, scoring=rmse_score, return_train_score=False) #error_score='raise', ,  verbose=True
model=clf.fit(X,y)


print(model.best_score_)
pickle._dump(model, open('Models/Total_Regression.sav', 'wb'))
rmtree(cachedir)





from sklearn.model_selection import cross_val_score


pipe = Pipeline(steps = [('reducer', ReduceCardinality()), ('featurize', FunctionFeaturizer()) , ('impute',DataFrameImputer()), (('positive', PositiveEstimator(RF))), ('classify', GroupEstimator(LGBMRegressor(n_jobs=4)))])

model = pipe.fit(X,y)

prds = model.predict(X)

performance_on_user(y, prds)


from sklearn.model_selection import KFold

kf = KFold(n_splits=5)

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    a = ReduceCardinality().fit_transform(X_train,y_train)
    print('done a')
    b = FunctionFeaturizer().fit_transform(a, y_train)
    print('done b')
    c = DataFrameImputer().fit_transform(b, y_train)
    print('done c')
    d=PositiveEstimator(RF).fit_transform(c, y_train)
    print('done d')
    e = GroupEstimator(LGBMRegressor(n_jobs=4)).fit(d, y_train).predict(d)
    print('done e')




    print('FITTED')
    prds = pipe.predict(X_test)
    print('PREDICTED')
    print(performance_on_user(y_true=y_test, y_pred=prds))

X.shape
y.shape

train_index.shape

X[test_index]
X.head
a

type(X)



a.dtypes


type(c)