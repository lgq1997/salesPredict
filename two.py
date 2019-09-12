#二次指数平滑对线性的预测真的好准啊，不管alpha取什么值结果都挺准的

from one import one_smooth
import matplotlib.pyplot as plt

def two_smooth(alpha,data):
    s_one = one_smooth(alpha,data) #一次指数平滑值
    s_two = one_smooth(alpha,s_one) #二次指数平滑值

    a_two = [i for i in range(0,len(data))]
    b_two = [i for i in range(0,len(data))]

    for i in range(len(data)):
        a_two[i] = 2*s_one[i] - s_two[i]
        b_two[i] = (alpha/(1-alpha))*(s_one[i] - s_two[i])

    return a_two,b_two


if __name__ == '__main__':
    alpha = 0.9
    #data = [i for i in range(100)]
    data = [133, 88, 150, 123, 404, 107, 674, 403, 243, 257, 900, 1043, 1156, 895,1200,1038,1024]
    a,b = two_smooth(alpha,data)
    for i in range(1,5):
        data_ = a[-1] + b[-1]*i
        print(data_)
        data.append(data_)
    print(data)

    plt.plot(data)
    plt.show()





