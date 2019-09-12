import pandas as pd
import matplotlib.pyplot as plt
import math

def getData():
    datas = pd.read_csv('train.csv', index_col='Date', parse_dates=['Date'])
    store1 = datas.loc[datas['Store'].isin([1])]
    store1['StateHoliday'].replace({'a': int(1), 'b': int(2), 'c': int(3), '0': int(0)}, inplace=True)
    store1 = store1.sort_index()
    train = store1.ix['2013-01-01':'2015-06-01']
    test = store1.ix['2015-06-02':'2015-07-31']
    X_train = train.drop(labels=['Sales','Customers'], axis=1)  # 删除Sales列，axis默认为0，是删除行
    Y_train = train['Sales']
    X_test = test.drop(labels=['Sales','Customers'], axis=1)
    Y_test = test['Sales']

    return X_train,Y_train,X_test,Y_test


def plot(Y_test,predict,title):
    fig = plt.figure(figsize=(8,5))
    plt.plot(predict,c = 'Red',marker = '*')
    plt.plot(Y_test,c = 'Black',marker = '*')
    plt.legend(['Predict data','Original data'])
    plt.xlabel('date')
    plt.ylabel('sales')
    plt.title(title)
    fig.autofmt_xdate()

def RMSPE(Y_test,predict):
    n = 0
    sum = 0
    for i in range(len(Y_test)):
        if Y_test[i] != 0:
            sum_y = ((Y_test[i] - predict[i])/Y_test[i])**2
            sum = sum + sum_y
            n = n+1
    rmspe = math.sqrt(sum/n)
    return rmspe


def RMSE(Y_test,predict):
    n = 0
    sum = 0
    for i in range(len(Y_test)):
        sum_y = ((Y_test[i] - predict[i])) ** 2
        sum = sum + sum_y
        n = n + 1
    rmse = math.sqrt(sum / n)
    return rmse




