import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv')
datas = df.loc[df['Store'].isin([1])] #提取店铺id为1的店铺信息
datas['StateHoliday'].replace({'a':int(1),'b':int(2),'c':int(3),'0':int(0)},inplace=True)
#datas942行9列
datas_y1 = datas.ix[:,'Sales'].values
datas_y = datas_y1[:,np.newaxis]  #把一维数组datas_y变为二维数组
datas_x = datas.drop(labels = ['Store','Date','Customers','Sales'],axis = 1).values


rnn_unit = 10  #隐藏层单元
input_size = 5  #输入层单元
output_size = 1  #输出层
lr = 0.0006  #学习率
tf.reset_default_graph()
#初始化输入层、输出层权重、偏置
weights = {
    'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),  #5行10列
    'out':tf.Variable(tf.random_normal([rnn_unit,1]))   #10行1列
}

biases = {
    'in':tf.Variable(tf.constant(0.1,shape = [rnn_unit,])), #表示一位数组，有rnn_unit个元素
    'out':tf.Variable(tf.constant(0.1,shape = [1,]))
}


def get_data(batch_size=60, time_step=15, train_begin=0, train_end = 897):
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

    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
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

'''
batch_index, train_x, train_y, test_x, test_y, scaler_for_y = get_data()
#print(train_x)
print(test_y)
#print(np.array(train_x).shape)
print(np.array(test_y).shape)
for i in test_y:
    print(np.array(i).shape)

'''


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
def train_lstm(batch_size=80, time_step=15, train_begin=0, train_end=897):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y, test_x, test_y, scaler_for_y = get_data(batch_size, time_step, train_begin,train_end)
    pred,final_states = lstm(X)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 重复训练5000次
        iter_time = 1000
        for i in range(iter_time):
            for step in range(len(batch_index) - 1):
                final_states,loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],Y: train_y[batch_index[step]:batch_index[step + 1]]})

            if i % 100 == 0:
                print('iter:', i, 'loss:', loss_)
        ####predict####
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
    return test_predict



test_predict = train_lstm(batch_size=80,time_step=15,train_begin=0,train_end=897)
#print(test_predict)

b = []
for item in test_predict:
    for i in item:
        if i < 0:
            i = 0
        b.append(i)

test_predict = b

plt.figure(figsize=(24,8))
plt.plot(datas_y1[897:])
#plt.plot(test_predict)
#plt.plot([None for i in range(897)] + [x for x in test_predict])
plt.plot(test_predict)
plt.show()




