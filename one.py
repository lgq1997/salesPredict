import matplotlib.pyplot as plt

#这个函数用于二次平滑和三次平滑
def one_smooth(alpha,y):
    y_ = [i for i in range(0,len(y))] #预测的y*，先占位
    y_[0] = (y[0]+y[1]+y[2])/3
    for i in range(1,len(y)):
        y_[i] = alpha*y[i] + (1-alpha)*y_[i-1]
    return y_






def one_smooth1(alpha,y):
    #print(y)
    y_ = [i for i in range(0,len(y))] #预测的y*，先占位
    y_[0] = (y[0]+y[1]+y[2])/3
    for i in range(1,len(y)):
        y_[i] = alpha*y[i-1] + (1-alpha)*y_[i-1]
    return y_

if __name__ == '__main__':
    alpha = 0.9  #对于拟合直线型，alpha对结果的影响挺大的
    date = [i for i in range(0, 100)]
    #one = one_smooth(alpha,date)
    one = one_smooth1(alpha,date)
    y_predict = alpha*date[-1] + (1-alpha)*one[-1]
    print(y_predict)
    date.append(y_predict)
    plt.plot(date)
    plt.show()







