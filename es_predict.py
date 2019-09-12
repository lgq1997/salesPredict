import matplotlib.pyplot as plt
import math
from dataset import getData,plot,RMSPE
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

#数据处理
X_train,Y_train,X_test,Y_test = getData()
train = Y_train
test = Y_test

def one_smooth(alpha,y):
    y_ = [i for i in range(0,len(y))] #预测的y*，先占位
    y_[0] = (y[0]+y[1]+y[2])/3
    for i in range(1,len(y)):
        y_[i] = alpha*y[i] + (1-alpha)*y_[i-1]
    return y_


def three_smooth(alpha,data):
    s_one = one_smooth(alpha,data)
    s_two = one_smooth(alpha,s_one)
    s_three = one_smooth(alpha,s_two)

    a_three = [i for i in range(len(data))]
    b_three = [i for i in range(len(data))]
    c_three = [i for i in range(len(data))]

    for i in range(len(data)):
        a_three[i] = 3*s_one[i] - 3*s_two[i] + s_three[i]
        b_three[i] = (alpha/(2*((1-alpha)**2)))*((6-5*alpha)*s_one[i] - 2*(5-4*alpha)*s_two[i] + (4-3*alpha)*s_three[i])
        c_three[i] = ((alpha**2)/(2*((1-alpha)**2))) * (s_one[i] - 2*s_two[i] + s_three[i])

    return a_three,b_three,c_three


if __name__ == '__main__':
#0.06  0.17078593793136068
#0.05  0.19104665873372706
#0.07  0.18185077499029648
    alpha = 0.06
    #data = [i for i in range(100)]
    data = train.values
    a,b,c = three_smooth(alpha,data)
    predict = []
    for i in range(1,len(test)+1):
        data_ = a[-1] + b[-1]*i + c[-1]*(i**2)
        print(data_)
        predict.append(data_)


    open = X_test['Open'].values
    for i in range(len(open)):
        if open[i] == 0:
            predict[i] = 0

    predict = pd.DataFrame({'Date': pd.date_range(start='2015-06-02', end='2015-07-31'), 'Sales': predict})
    predict = predict.set_index('Date')

    rmspe = RMSPE(test.values, predict.values)
    print(rmspe)

    plot(test,predict,'Exponential Smoothing')
    plt.show()

#0.17078593793136068


