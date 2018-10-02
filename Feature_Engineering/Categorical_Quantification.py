from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np



class FunctionFeaturizer(BaseEstimator, TransformerMixin):
    '''
    This one is for mean - it can be used for binary or numerical class
    In case of binary it is interpreted as ratio of positive class
    '''
    REQUIREMENTS = [
        'pandas',  # install specific version of a package
        'scikit-learn',  # install latest version of a package
        'numpy'  # install latest version of a package
    ]

    def __init__(self):
        self.featurizers = None
        #self.pd = pd

    def fit(self, X, y=None):
        '''
        Creates map between values of categorical features and corresponding quantifications
        :param X:
        :param y:
        :return:
        '''
        featurizers = []
        categorical_features = list(X.select_dtypes(include='object').columns)
        for att in categorical_features:
            quantified = pd.concat([X[att], y], axis=1)
            grouped = quantified.groupby(att).agg('mean') # this one is aggregation it can be change with other methods
            grouped.columns = [att]
            featurizers.append(grouped)
        self.featurizers=featurizers
        return self

    def transform(self, X):
        #Do transformations
        for feat in self.featurizers:
            att = feat.columns[0]
            X = X.join(feat, att, rsuffix='_dj_oka' + att, how='left')
            X = X.drop(att, axis=1)
        col_names = X.columns
        new_names = [name.split('_dj_oka')[1] if len(name.split('_dj_oka')) > 1 else name for name in col_names]
        X.columns = new_names
        return X


class ReduceCardinality(BaseEstimator, TransformerMixin):
    '''
        This one is for mean - it can be used for binary or numerical class
        In case of binary it is interpreted as ratio of positive class
        '''
    REQUIREMENTS = [
        'pandas',  # install specific version of a package
        'scikit-learn',  # install latest version of a package
        'numpy'  # install latest version of a package
    ]

    def __init__(self, threshold=5):
        self.category_map = {}
        self.threshold = threshold

    def fit(self, X, y=None):
        atts = X.columns
        count_dict = self.num_of_occurrence_in_cat(X)

        for key in count_dict.keys():
            if ((key in atts) and (key not in ['fullVisitorId', 'sessionId', 'gclId'])):
                percentage = (count_dict[key] / np.sum(count_dict[key]) * 100).reset_index()
                low_percentage_values = percentage[percentage[key] < self.threshold]['index'].tolist()
                percentage['map'] = percentage['index'].apply(lambda x: 'other' if x in low_percentage_values else x)
                percentage = percentage.drop(key, axis=1)
                percentage.columns = [key, 'map']
                self.category_map.update({key: percentage})
        return self

    def transform(self, X):
        for key, val in self.category_map.items():
            X = X.merge(val, on=key, how='left')
            X = X.drop(key, axis=1)
            X = X.rename(columns={'map': key})
        return X

    def num_of_occurrence_in_cat(self, X):

        logical_vec = (X.dtypes == object)
        result_dict = {}  # pd.DataFrame(columns=['att_name','category','count'])

        for i in range(len(logical_vec)):

            if logical_vec[i] == True:
                att = X.columns[i]
                count_val = pd.DataFrame(pd.value_counts(X.iloc[:, i]))
                result_dict.update({att: count_val})

        return result_dict


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


