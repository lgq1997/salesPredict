from one import one_smooth
import matplotlib.pyplot as plt
import math

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

    alpha = 0.3
    #data = [i for i in range(100)]
    data = [133,88,150,123,404,107,674,403,243,257,900,1043,1156,895]
    a,b,c = three_smooth(alpha,data)
    for i in range(1,15):
        data_ = a[-1] + b[-1]*i + c[-1]*(i**2)
        print(data_)
        data.append(data_)
    print(data)

    plt.plot(data)
    plt.show()