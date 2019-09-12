import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from dataset import getData,plot,RMSPE,RMSE
import warnings
import time

start = time.clock()
warnings.filterwarnings('ignore')


df = pd.read_csv('train.csv',index_col='Date', parse_dates=['Date'])
datas = df.loc[df['Store'].isin([1])] #提取店铺id为1的店铺信息
datas['StateHoliday'].replace({'a':int(1),'b':int(2),'c':int(3),'0':int(0)},inplace=True)
#datas942行9列
datas = datas.sort_index()

train = datas.ix['2013-01-01':'2015-06-01']
test = datas.ix['2015-06-02':'2015-07-31']

datas_y1 = train.ix[:,'Sales'].values

datas_y = datas_y1[:,np.newaxis]  #把一维数组datas_y变为二维数组
datas_x = train.drop(labels = ['Store','Customers','Sales'],axis = 1).values
#9   0.1593714551437923
#10  0.11174295165692139
#11  0.19844746594437318
rnn_unit = 10  #隐藏层单元
input_size = 5  #输入层单元
output_size = 1  #输出层

#0.05    0.1471511161350663
#0.06    0.12067445130819968
#0.065   0.11174295165692139
#0.07    0.1357708336560025
lr = 0.065  #学习率
tf.reset_default_graph()
#初始化输入层、输出层权重、偏置
weights = {
    'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),  #5行10列  ,正态分布
    'out':tf.Variable(tf.random_normal([rnn_unit,1]))   #10行1列
}

biases = {
    'in':tf.Variable(tf.constant(0.1,shape = [rnn_unit,])), #表示一位数组，有rnn_unit个元素  ，都设为0.1
    'out':tf.Variable(tf.constant(0.1,shape = [1,]))
}


def get_data(batch_size=80, time_step=15, train_begin=0, train_end = 762):
    batch_index = []
    scaler_for_x = MinMaxScaler(feature_range=(0, 1))  # 做minmax缩放
    scaler_for_y = MinMaxScaler(feature_range=(0, 1))
    scaled_x_data = scaler_for_x.fit_transform(datas_x)
    scaled_y_data = scaler_for_y.fit_transform(datas_y)
    label_train = scaled_y_data[train_begin:train_end]
    label_test = scaled_y_data[train_end:]
    normalized_train_data = scaled_x_data[train_begin:train_end]
    normalized_test_data = scaled_x_data[train_end:]

    train_x, train_y = [], []  # 训练集x和y初定义
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :5]
        y = label_train[i:i + time_step]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))

    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample,//表示地板除
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :5]
        y = label_test[i * time_step:(i + 1) * time_step]
        test_x.append(x.tolist())
        #test_y.extend(y)
        test_y.append(y.tolist())
    test_x.append((normalized_test_data[(i + 1) * time_step:, :5]).tolist())
    test_y.append((label_test[(i + 1) * time_step:]).tolist())

    return batch_index, train_x, train_y, test_x, test_y, scaler_for_y

#——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit)
    #cell=tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



# ——————————————————训练模型——————————————————
def train_lstm(batch_size=80, time_step=15, train_begin=0, train_end=763):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y, test_x, test_y, scaler_for_y = get_data(batch_size, time_step, train_begin,train_end)
    pred,final_states = lstm(X)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    xx = []
    yy = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 重复训练200次
        iter_time = 200
        for i in range(iter_time):
            for step in range(len(batch_index) - 1):
                final_states,loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],Y: train_y[batch_index[step]:batch_index[step + 1]]})

            if i % 100 == 0:
                print('iter:', i, 'loss:', loss_)
                xx.append(i)
                yy.append(loss_)
        ####预测####
        test_predict = []
        for step in range(len(test_x)):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            #test_predict.extend(predict)
            test_predict.append(predict)

        test_predict = scaler_for_y.inverse_transform(test_predict)
        #test_y = scaler_for_y.inverse_transform(test_y)
        a = []
        for i in test_y:
            a.append(scaler_for_y.inverse_transform(i))
        test_y = a
        #rmse = np.sqrt(mean_squared_error(test_predict,test_y))
        #mae = mean_absolute_error(y_pred=test_predict, y_true=test_y)
        #print('mae:', mae, '   rmse:', rmse)
    return test_predict,xx,yy


test_predict,xx,yy = train_lstm(batch_size=80,time_step=15,train_begin=0,train_end=762)
predict = []
for i in range(4):
    for j in test_predict[i]:
        predict.append(j)


X_train1,Y_train1,X_test1,Y_test1 = getData()
'''
open = X_test1['Open'].values
for i in range(len(open)):
    if open[i] == 0:
        predict[i] = 0
'''
print(predict)
predict = pd.DataFrame({'Date':pd.date_range(start='2015-06-02',end='2015-07-31'),'Sales':predict})
predict = predict.set_index('Date')


rmspe = RMSPE(Y_test1.values,predict.values)
print(rmspe)

rmse = RMSE(Y_test1.values,predict.values)
print(rmse)

end = time.clock()
print(end-start)

plot(Y_test1,predict,'LSTM')
plt.show()

#428.28796613950356
#343.9435204481141


'''
1
[5019.581887722015, 4891.921279907227, 0, 5671.705658912659, 4844.845830917358, 0, 3740.4054350852966, 3451.8879976272583, 3555.696825027466, 3867.4049921035767, 4198.802988767624, 4507.455499649048, 0, 5527.784564495087, 5086.647876262665, 4820.7254276275635, 4825.269299983978, 4878.375417709351, 4832.083688735962, 0, 3710.273388147354, 3444.6300687789917, 3554.7367680072784, 3867.215877056122, 4198.791346549988, 4507.450388431549, 0, 5527.749921798706, 5086.660938262939, 4975.25513792038, 4713.514233112335, 4834.786954879761, 4850.331871032715, 0, 3716.425590276718, 3445.773277759552, 3555.153332233429, 3867.3161137104034, 4198.8132112026215, 4507.4529440402985, 0, 5527.702785015106, 5086.679111480713, 4975.256841659546, 4939.365872383118, 4689.209825992584, 4847.348623752594, 0, 3731.2830476760864, 3448.6128430366516, 3556.1738719940186, 3867.571106672287, 4198.865175247192, 4507.460042953491, 0, 6037.55922460556, 5352.603831768036, 5144.923138618469, 5100.973483085632, 5050.115165233612]
0.10659407121826824
413.4215011547876
'''

'''
2
[5082.044373035431, 4793.210040092468, 0, 5907.198758125305, 4729.922093153, 0, 3566.0745842456818, 3338.314189195633, 3384.8671581745148, 3624.2464864254, 3909.8462705612183, 4661.027708530426, 0, 5414.186619758606, 4941.575622081757, 5000.626653671265, 4693.775563001633, 4640.682791233063, 4738.577656030655, 0, 3574.730147123337, 3342.1711707115173, 3386.5328471660614, 3623.9807031154633, 3909.675044775009, 4661.210860490799, 0, 5414.522824287415, 4941.6119685173035, 4828.74719953537, 4956.125554561615, 4622.925001859665, 4719.21125292778, 0, 3586.2729799747467, 3346.790575504303, 3388.5983469486237, 3624.2567088603973, 3909.686970949173, 4661.288948535919, 0, 5414.690926551819, 4941.636956691742, 4828.763669013977, 4723.245707273483, 4924.634207725525, 4573.624185562134, 0, 3615.7860016822815, 3356.8485996723175, 3393.0465259552, 3624.8578448295593, 3909.7196259498596, 4661.441433191299, 0, 6278.1357135772705, 5668.304995536804, 5475.157765388489, 5206.541705131531, 4911.622751712799]
0.10843829763715078
429.6108290851146
'''

