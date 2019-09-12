import matplotlib.pyplot as plt
from dataset import getData,plot,RMSPE,RMSE
import pandas as pd
import warnings
import numpy as np

warnings.filterwarnings('ignore')


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


#数据处理
X_train,Y_train,X_test,Y_test = getData()
train = Y_train.values  #882
test = Y_test

train1 = train[0:440]  #0.03  0.17  0.01
test1 = train[441:500]  #
train2 = train[100:540]  #
test2 = train[541:600]
train3 = train[200:640]
test3 = train[641:700]
train4 = train[300:740]
test4 = train[741:800]
train5 = train[400:822]
test5 = train[823:882]

data = [[1,train1,test1],[2,train2,test2],[3,train3,test3],[4,train4,test4],[5,train5,test5]]
trainall = [train1,train2,train3,train4,train5]
testall = [test1,test2,test3,test4,test5]
'''
train1 = train[0:115]  #0.45  0.05  0.01    0.05, 0.65, 0.01
test1 = train[116:176]  #0.1  0.03  0.04
train2 = train[177:293]  #0.1
test2 = train[294:354]
train3 = train[355:471]
test3 = train[472:532]
train4 = train[533:649]
test4 = train[650:710]
train5 = train[711:827]
test5 = train[828:882]
'''


def hw(train,alpha,beta,gamma):
    s = [i for i in range(len(train) + 1)]
    b = [i for i in range(len(train) + 1)]
    c = [i for i in range(len(train) + 15)]

    s[0] = train[0]
    b[0] = train[1] - train[0]
    c[8] = 0


    for i in range(1, len(train) + 1):
        s[i] = alpha * (train[i - 1] - c[i + 7]) + (1 - alpha) * (s[i - 1] + b[i - 1])
        b[i] = beta * (s[i] - s[i - 1]) + (1 - beta) * b[i - 1]
        c[i + 14] = gamma * (train[i - 1] - s[i - 1] - b[i - 1]) + (1 - gamma) * c[i + 7]

    predict = []
    for i in range(1, len(test) + 1):
        # f_ = s[-1] + i*b[-1] + c[len(c)-7+((i-1)%7)]
        f_ = s[-1] + i * b[-1] + c[len(c) - 7 + np.mod(i - 1, 7)]
        predict.append(f_)

    return predict

def canshu(train,test):
    ll = []
    aa = []
    for i in np.arange(0.01,0.99,0.02):
        for j in np.arange(0.01,0.99,0.02):
            for k in np.arange(0.01,0.99,0.02):
                predict = hw(train,i,j,k)
                rmspe = RMSPE(test, predict)
                ll.append(rmspe)
                aa.append([rmspe,i,j,k])
    a = ll.index(min(ll))
    return aa[a]

for index,traina,testa in data:
    shuju = canshu(traina,testa)
    print('index:',index,'alpha:',shuju[1],'beta:',shuju[2],'gamma:',shuju[3],'rmspe:',shuju[0])





#[0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,0.8, 0.85, 0.9, 0.95]
'''
p1 = hw(train1,0.45,0.05,0.01)
p2 = hw(train1,0.05,0.65,0.01)
r1 = RMSPE(test1,p1)
r2 = RMSPE(test1,p2)
print(r1,r2)
'''







