from sklearn.model_selection import GridSearchCV
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.svm import SVR
from dataset import getData,plot,RMSPE,RMSE
import numpy as np
import time

start = time.clock()
warnings.filterwarnings('ignore')
X_train,Y_train,X_test,Y_test = getData()

#X_train.shape (882, 6)
#Y_train.shape (882,)

print(X_train)
print(Y_train)

'''
svr = GridSearchCV(SVR(), param_grid={"kernel": ("linear", 'rbf'), "C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)})

svr.fit(X_train, Y_train)



y_predict = svr.predict(X_test)

open = X_test['Open'].values
for i in range(len(open)):
    if open[i] == 0:
        y_predict[i] = 0

print(y_predict.tolist())

Y_predict = pd.DataFrame({'Date':pd.date_range(start='2015-06-02',end='2015-07-31'),'Sales':y_predict})
Y_predict = Y_predict.set_index('Date')

rmspe = RMSPE(Y_test.values,Y_predict.values)
print(rmspe)

rmse = RMSE(Y_test.values,Y_predict.values)
print(rmse)

end = time.clock()
print(end-start)

plot(Y_test,Y_predict,'svr')
plt.show()

'''

#0.12447836495827969
#477.7168718919134
#23.16800472815463
#参数的最佳取值：{'C': 1000.0, 'gamma': 1.0, 'kernel': 'rbf'}