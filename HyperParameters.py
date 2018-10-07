'''
param_grid = {
    'classify__n_estimators':[100, 200, 300, 500],
    'classify__max_depth':[3, 5, 7, 10]
}

'''

param_grid_XGboost = {
    'classify__base_estimator__learning_rate': [0.3],
    'classify__base_estimator__gamma': [0.01, 0.05, 0.1, 0.3, 0.5, 1, 5],
    'classify__base_estimator__max_depth': [3, 5, 10, 15, 20],
    'classify__base_estimator__colsample_bytree': [0.1, 0.5, 0.9],
    'classify__base_estimator__colsample_bylevel': [0.1, 0.5, 0.9],
    'classify__base_estimator__reg_lambda': [0.1, 0.3, 0.5, 0.9],
    'classify__base_estimator__reg_alpha': [0.1, 0.3, 0.5, 0.9],
    'classify__base_estimator__subsample': [0.7, 1],
    'classify__base_estimator__n_estimators': [50, 100, 500, 1000, 3000]
}

'''
base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=nan, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1). Check the list of available parameters with `estimator.get_params().keys()`.
'''

'''
param_grid_XGboost ={
    'classify__base_estimator__gamma':[0, 0.3],
    'classify__base_estimator__max_depth':[5],
    'classify__base_estimator__n_estimators':[20],

}
'''


param_grid_LGBM = {
    'classify__base_estimator__objective':['regression'],
    'classify__base_estimator__boosting_type':['gbdt'],
    'classify__base_estimator__metric':['rmse'],
    'classify__base_estimator__n_estimators':[100, 500],#, 300, 500, 1000, 5000], #10000
    'classify__base_estimator__num_leaves':[70],
    'classify__base_estimator__learning_rate':[0.1], #0.01
    'classify__base_estimator__bagging_fraction':[0.7],#0.8
    'classify__base_estimator__feature_fraction':[0.7],#.3
    'classify__base_estimator__max_depth':[5, 10, 15] #-1
}


param_grid_XGboost ={
    'classify__base_estimator__learning_rate':[0.3],
    'classify__base_estimator__gamma':[0.01, 0.05, 0.1],
    'classify__base_estimator__max_depth':[5,10,15],
    'classify__base_estimator__colsample_bytree':[0.7,1],
    'classify__base_estimator__colsample_bylevel':[0.7,1],
    'classify__base_estimator__reg_lambda': [0.1],
    'classify__base_estimator__reg_alpha': [0.1, 0.5],
    'classify__base_estimator__subsample': [0.7, 1],
    'classify__base_estimator__n_estimators':[50, 100, 500]
}
