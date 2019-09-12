import matplotlib.pyplot as plt
from dataset import getData,plot,RMSPE,RMSE
import pandas as pd
import numpy as np
import warnings
import time
from sklearn.model_selection import GridSearchCV

start = time.clock()
warnings.filterwarnings('ignore')

#数据处理
X_train,Y_train,X_test,Y_test = getData()
train = Y_train.values
test = Y_test

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
        f_ = s[-1] + i * b[-1] + c[len(c) - 7 + np.mod(i - 1, 7)]
        predict.append(f_)
    return predict



predict = hw(train,0.05,0.15,0.01)

open = X_test['Open'].values
for i in range(len(open)):
    if open[i] == 0:
        predict[i] = 0


print(predict)
predict = pd.DataFrame({'Date': pd.date_range(start='2015-06-02', end='2015-07-31'), 'Sales': predict})
predict = predict.set_index('Date')


rmspe = RMSPE(test.values, predict.values)
print(rmspe)

rmse = RMSE(test.values, predict.values)
print(rmse)

end = time.clock()
print(end-start)

plot(test,predict,'holt winters')
plt.show()



'''
[4078.688909737665, 3914.9182155556387, 0, 3878.2511613264446, 4278.099628828576, 0, 4273.686082464293, 4078.9217627910157, 3915.151068608989, 3628.0536168821154, 3878.484014379795, 4278.332481881926, 0, 4273.918935517643, 4079.154615844366, 3915.3839216623396, 3628.286469935466, 3878.7168674331456, 4278.565334935276, 0, 4274.151788570994, 4079.3874688977157, 3915.61677471569, 3628.5193229888164, 3878.949720486496, 4278.7981879886265, 0, 4274.384641624344, 4079.620321951066, 3915.8496277690406, 3628.752176042167, 3879.1825735398456, 4279.031041041977, 0, 4274.617494677695, 4079.8531750044167, 3916.082480822391, 3628.9850290955173, 3879.415426593196, 4279.2638940953275, 0, 4274.850347731044, 4080.086028057767, 3916.3153338757415, 3629.217882148868, 3879.6482796465466, 4279.496747148678, 0, 4275.083200784395, 4080.3188811111177, 3916.548186929092, 3629.4507352022174, 3879.881132699897, 4279.729600202028, 0, 4275.316053837745, 4080.551734164468, 3916.7810399824425, 3629.683588255568, 3880.1139857532476]
0.16513565625542484
776.7568224130538
1.2834301828308678
'''