from dataset import getData,plot,RMSPE,RMSE
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import warnings
import time
import numpy as np

start = time.clock()
warnings.filterwarnings('ignore')

X_train,Y_train,X_test,Y_test = getData()
train = Y_train
test = Y_test

#差分查看平稳性

train_diff1 = train.diff(1)
train_diff2 = train_diff1.diff(1)
train_diff1.dropna(inplace=True)

#ADF检验
#(-11.052828250801923, 5.029832778965925e-20, 20, 860, {'1%': -3.4379766581448803, '5%': -2.8649066016199836, '10%': -2.5685626352082207}, 14694.285579031513)
'''
train_diff1 = train_diff1.tolist()
print(sm.tsa.stattools.adfuller(train_diff1))
'''
'''
plt.subplot(2,1,1)
plt.plot(train_diff1)
plt.title('one diff')
plt.subplot(2,1,2)
plt.plot(train_diff2)
plt.title('two diff')

plt.show()
'''

#查看ACF 和 PACF
'''
fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train_diff1, lags=100, ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout()

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train_diff1, lags=100, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout()
plt.show()
'''

#计算AIC和BIC
'''
train_results = sm.tsa.arma_order_select_ic(train_diff1, ic=['aic', 'bic'], trend='nc', max_ar=15, max_ma=15)
print('AIC', train_results.aic_min_order)
print('BIC', train_results.bic_min_order)
'''



model = sm.tsa.ARIMA(train, order=(13, 1, 0)) #（p,d,q）
results = model.fit()
predict = results.forecast(len(test))[0]

open = X_test['Open'].values
for i in range(len(open)):
    if open[i] == 0:
        predict[i] = 0

print(predict.tolist())
predict = pd.DataFrame({'Date':pd.date_range(start='2015-06-02',end='2015-07-31'),'Sales':predict})
predict = predict.set_index('Date')

rmspe = RMSPE(test.values,predict.values)
print(rmspe)

rmse = RMSE(test.values,predict.values)
print(rmse)

end = time.clock()
print(end-start)

plot(test,predict,'ARIMA')
plt.show()



#0.15629390036574742
#683.436169886969
#153.88056277698684

#模型检验
'''
model = sm.tsa.ARIMA(train, order=(13, 1, 0))
results = model.fit()
resid = results.resid #赋值
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=80)
plt.show()
'''


'''
[5380.2656548849, 4901.982786062458, 0.0, 4140.574427168025, 4788.929138149109, 0.0, 2596.3671958247633, 4213.928543065477, 3666.053234903123, 3685.55596777421, 4306.2000997166215, 5038.16500804933, 0.0, 4572.674508168282, 5097.69283008072, 4531.09305359815, 4346.834493076916, 4237.755650351872, 4858.9683853101415, 0.0, 3433.8001923819315, 4317.604044914595, 3601.881620703172, 3659.527009961887, 4062.190460260721, 4755.089146696046, 0.0, 4186.748402810273, 4829.032445889124, 4167.4699791590465, 4139.302987834669, 4221.900204184452, 4781.481972654258, 0.0, 3764.7199310729848, 4376.604998932266, 3596.9923588038037, 3685.8707268863545, 3967.9515149258264, 4583.97071350642, 0.0, 4041.52773270588, 4623.235337221083, 3906.658478269645, 3985.307805981153, 4148.868541086721, 4655.316157971694, 0.0, 3902.8612551767956, 4384.04989639678, 3591.185588159993, 3712.6164586137306, 3938.1584618083166, 4463.7979476996, 0.0, 3982.483347988852, 4473.699365852642, 3739.531176630887, 3883.483424720366, 4071.3686680603932]
0.16005149455000875
695.6148711898246
32.614690532070675
'''