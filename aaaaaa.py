import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from dataset import getData,plot,RMSPE,RMSE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import warnings
import time

start = time.clock()
warnings.filterwarnings('ignore')
RMSPE_score = make_scorer(RMSPE, greater_is_better = False)
X_train,Y_train,X_test,Y_test = getData()

#n_estimators 最佳值550
'''
cv_params = {'n_estimators':[400,450,500,550,600,650,700,750,800,850,900,950,1000]}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
'''

#参数的最佳取值：{'max_depth': 2, 'min_child_weight': 9}
'''
cv_params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9,10]}
other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
'''

#参数的最佳取值：{'gamma': 0.2}
'''
cv_params = {'gamma': [0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 2, 'min_child_weight': 9, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
'''



#参数的最佳取值：{'colsample_bytree': 0.4, 'subsample': 0.9}
'''
cv_params = {'subsample': [0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1], 'colsample_bytree': [0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]}
other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 2, 'min_child_weight': 9, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 0, 'reg_lambda': 1}
'''

#参数的最佳取值：{'reg_alpha': 0.02, 'reg_lambda': 1}
'''
cv_params = {'reg_alpha': [0.02, 0.05, 0.1, 0.5, 1], 'reg_lambda': [0.02, 0.05, 0.1, 0.5, 1, 2]}
other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 2, 'min_child_weight': 9, 'seed': 0,
                'subsample': 0.9, 'colsample_bytree': 0.4, 'gamma': 0.2, 'reg_alpha': 0, 'reg_lambda': 1}
'''

#参数的最佳取值：{'learning_rate': 0.1}
'''
cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
other_params = {'learning_rate': 0.07, 'n_estimators': 550, 'max_depth': 2, 'min_child_weight': 9, 'seed': 0,
                'subsample': 0.9, 'colsample_bytree': 0.4, 'gamma': 0.2, 'reg_alpha': 0.02, 'reg_lambda': 1}

'''
'''
model = xgb.XGBRegressor(**other_params)
#optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring=RMSPE_score, cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, Y_train)

evalute_result = optimized_GBM.cv_results_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
'''
'''
#learning_rate 0.06  0.13491557942089408
#0.05  0.1337607555654951
#0.04  0.1327764588427787
#0.03  0.13166466643402547
#0.02  0.1305988764992404
#0.01  0.12965518680990745
#0.005  0.1059106953997303
#0.006  0.11443387836421556
'''
'''
other_params = {'learning_rate': 0.1, 'n_estimators': 550, 'max_depth': 2, 'min_child_weight': 9, 'seed': 0,
                'subsample': 0.9, 'colsample_bytree': 0.4, 'gamma': 0.2, 'reg_alpha': 0.02, 'reg_lambda': 1}
'''
'''
#600  0.11910446587586643
#550  0.11443387836421556
#500  0.1093268927294404
#400  0.10779259608543522

#max depth 2  0.10779259608543522
#3  0.10171728077426573

#'min_child_weight' 9 0.10171728077426573
#8  0.10139469350522683
#7  0.10134878506034584
#5  0.10130818327391648

#seed 2  0.09881332139579216
#3  0.09784835041126878
#4  0.09735304970964129
#5  0.09925829501074168

#0.8  0.09734716616831283

#0.5  0.09209990198701486

#0.5  0.09177947386717084
#0.1 0.09163980031597718
#0.05  0.09164398223795756
'''
'''
other_params = {'learning_rate': 0.006,
                'n_estimators': 400,
                'max_depth': 3,
                'min_child_weight': 5,
                'seed': 4,
                'subsample': 0.8,
                'colsample_bytree': 0.5,
                'gamma': 0.2,
                'reg_alpha': 0.02,
                'reg_lambda': 0.1}
'''

other_params = {'learning_rate': 0.01,
                'n_estimators': 450,
                'max_depth': 3,
                'min_child_weight': 2,
                'seed': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.4,
                'gamma': 0.2,
                'reg_alpha': 0.02,
                'reg_lambda': 0.1}


model = xgb.XGBRegressor(**other_params) #指定模型
model.fit(X_train, Y_train) #训练模型
predict = model.predict(X_test) #预测

open = X_test['Open'].values
for i in range(len(open)):
    if open[i] == 0:
        predict[i] = 0

print(predict.tolist())
predict = pd.DataFrame({'Date':pd.date_range(start='2015-06-02',end='2015-07-31'),'Sales':predict})
predict = predict.set_index('Date')

rmspe = RMSPE(Y_test.values,predict.values)
print(rmspe)

rmse = RMSE(Y_test.values,predict.values)
print(rmse)

end = time.clock()
print(end-start)

plot(Y_test,predict,'XGBoost')
plt.show()

#0.09163980031597718
#410.174619826727
#2.8563968433922797


