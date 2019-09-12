'''
from one import one_smooth
import matplotlib.pyplot as plt
from dataset import getData,plot,RMSPE
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

#数据处理
X_train,Y_train,X_test,Y_test = getData()
train = Y_train.values
test = Y_test


s = [i for i in range(len(train)+1)]
b = [i for i in range(len(train)+1)]
c = [i for i in range(len(train)+15)]

s[0] = train[0]
b[0] = train[1] - train[0]
c[8] = 1
#0.02  0.3173352645813184
#0.03  0.24817178084369876
#0.04  0.28502015654099966
alpha = 0.03

#0.05  0.24817178084369876
beta = 0.05
gamma = 0.01
for i in range(1,len(train)+1):

    s[i] = alpha*(train[i-1]/c[i+7]) + (1-alpha)*(s[i-1] + b[i-1])
    b[i] = beta*(s[i] - s[i-1]) + (1-beta)*b[i-1]
    c[i+14] = gamma*(train[i-1]/s[i]) + (1-gamma)*c[i+7]

predict = []
for i in range(1,len(test)+1):
    #f_ = s[-1] + i*b[-1] + c[len(c)-7+((i-1)%7)]
    f_ = (s[-1] + i * b[-1])*c[len(c) - 7 + np.mod(i-1,7)]
    predict.append(f_)

open = X_test['Open'].values
for i in range(len(open)):
    if open[i] == 0:
        predict[i] = 0

predict = pd.DataFrame({'Date': pd.date_range(start='2015-06-02', end='2015-07-31'), 'Sales': predict})
predict = predict.set_index('Date')

rmspe = RMSPE(test.values, predict.values)
print(rmspe)

plot(test,predict,'holt winters')
plt.show()

'''

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

#n_estimators 最佳值550    700
'''
cv_params = {'n_estimators':[400,450,500,550,600,650,700,750,800,850,900,950,1000]}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
'''

#参数的最佳取值：{'max_depth': 2, 'min_child_weight': 9}
# 2   1
'''
cv_params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6, 7]}
other_params = {'learning_rate': 0.1, 'n_estimators': 700, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
'''

#参数的最佳取值：{'gamma': 0.2}  0.005
'''
cv_params = {'gamma': [0.005,0.01,0.02,0.03,0.04,0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
other_params = {'learning_rate': 0.1, 'n_estimators': 700, 'max_depth': 2, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
'''



#参数的最佳取值：{'colsample_bytree': 0.4, 'subsample': 0.9}   0.4  0.6
'''
cv_params = {'subsample': [0.3,0.4, 0.5, 0.6, 0.7, 0.8], 'colsample_bytree': [0.3,0.4, 0.5, 0.6, 0.7, 0.8]}
other_params = {'learning_rate': 0.1, 'n_estimators': 700, 'max_depth': 2, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.005, 'reg_alpha': 0, 'reg_lambda': 1}
'''


#参数的最佳取值：{'reg_alpha': 0.02, 'reg_lambda': 1}   0.5  0.05

'''
cv_params = {'reg_alpha': [0.02, 0.05, 0.1, 0.5, 1], 'reg_lambda': [0.02, 0.05, 0.1, 0.5, 1, 2]}
other_params = {'learning_rate': 0.1, 'n_estimators': 700, 'max_depth': 2, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.6, 'colsample_bytree': 0.4, 'gamma': 0.005, 'reg_alpha': 0, 'reg_lambda': 1}
'''

#参数的最佳取值：{'learning_rate': 0.1}
'''
cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
other_params = {'learning_rate': 0.07, 'n_estimators': 700, 'max_depth': 2, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.6, 'colsample_bytree': 0.4, 'gamma': 0.005, 'reg_alpha': 0.05, 'reg_lambda': 0.05}
'''

model = xgb.XGBRegressor(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
#optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring=RMSPE_score, cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, Y_train)

evalute_result = optimized_GBM.cv_results_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))




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



