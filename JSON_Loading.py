from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator


class FunctionFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, *featurizers):
        self.featurizers = featurizers

    def fit(self, X, y=None):
        featurizers = []
        categorical_features = list(X.select_dtypes(include='object').columns)
        for att in categorical_features:
            quantified = pd.concat([X[att], y], axis=1)
            grouped = quantified.groupby(att).agg('mean')
            featurizers.append(grouped)
        self.featurizers = featurizers
        return self


    def transform(self, X):
        # Do transformations
        for feat in self.featurizers:
            att = feat.columns[0]
            X = X.join(feat, att, rsuffix='_dj_oka' + att)
            X = X.drop(att, axis=1)
        col_names = X.columns
        new_names = [name.split('_dj_oka')[1] if len(name.split('_dj_oka')) > 1 else name for name in col_names]
        X.columns = new_names
        return X
