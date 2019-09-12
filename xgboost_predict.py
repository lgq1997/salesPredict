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

#print(X_train.shape) #(882, 6)
#print(Y_train.shape) #(882,)


print(X_test)
print(Y_test)

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
'''



#0.09163980031597718
#410.174619826727
#2.8563968433922797


