import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100)


df = pd.read_csv('reduced_data.csv', low_memory=False)



'''
Basic prep - Extract VisitorId, y, and remove unused variables 
'''

visitor_id = df['fullVisitorId']
y_numerical = df['transactionRevenue']

remove_atts = ['Unnamed: 0', 'sessionId', 'visitId', 'visitStartTime', 'fullVisitorId', 'id_real', 'networkDomain', 'transactionRevenue']

y_numerical = y_numerical.apply(lambda x: 1 if x>0 else x)
y =y_numerical.astype('int')

y.value_counts()

df = df.drop(remove_atts, axis=1)

'''
Prepare categorical for submission
'''

def quantify_categorical(df, y):

    categorical_features = list(df.select_dtypes(include='object').columns)
    for att in categorical_features:
        quantified = pd.concat([df[att], y] , axis =1)
        grouped = quantified.groupby(att).agg('mean')
        df = df.join(grouped, att, rsuffix='_'+att)
    df = df.drop(categorical_features, axis=1)
    col_names = df.columns
    new_names = [name.split(y.name+'_')[1] if len(name.split(y.name+'_'))>1 else name for name in col_names ]
    df.columns = new_names
    return(df)

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


class FunctionFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.featurizers = None

    def fit(self, X, y=None):
        featurizers = []
        categorical_features = list(X.select_dtypes(include='object').columns)
        for att in categorical_features:
            quantified = pd.concat([X[att], y], axis=1)
            grouped = quantified.groupby(att).agg('mean')
            grouped.columns = [att]
            featurizers.append(grouped)
        self.featurizers=featurizers
        return self

    def transform(self, X):
        #Do transformations
        for feat in self.featurizers:
            att = feat.columns[0]
            X = X.join(feat, att, rsuffix='_dj_oka' + att)
            X = X.drop(att, axis=1)
        col_names = X.columns
        new_names = [name.split('_dj_oka')[1] if len(name.split('_dj_oka')) > 1 else name for name in col_names]
        X.columns = new_names
        return X

ff = FunctionFeaturizer()

a= ff.fit(df, y_numerical)
b=a.transform(df)
a.featurizers[0]
b.head()




from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix



X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, stratify=y)


pipe = Pipeline(steps = [('featurize', FunctionFeaturizer()), ('classify', LogisticRegression())])

param_grid = {
    'classify__class_weight':['balanced'],
    'classify__C':[0.01, 0.5]
}

clf = GridSearchCV(pipe, param_grid = param_grid, cv=4, scoring = 'roc_auc', n_jobs=4, return_train_score=False)

model=clf.fit(X=X_train, y=y_train)

model.cv_results_

predictions = model.predict(X_test)
print(roc_auc_score(y_test, predictions))


pred = np.column_stack((np.array(y_test), predictions))

confusion_matrix(y_test, predictions)


pred.shape

pd.DataFrame(pred)

type(y_test)
type(predictions)




df.head()


df_quantified = quantify_categorical(df, y_numerical)


LR = LogisticRegressionCV(class_weight='balance',  )



'''
Ideas for categorization

Chi2, Decision tree, avg (in case of binary)
Normalized avg ( max-min)
'''






'''
Binary classification - identify not 0 transactions
'''


# categorical to numerical variables





'''
Prepare for submission
'''

