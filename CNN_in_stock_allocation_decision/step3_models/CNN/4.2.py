import numpy as np
import pandas as pd
import progressbar
import time
import datetime
import parser
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.dates import AutoDateLocator, DateFormatter  
import seaborn as sns
import datetime
import os
import tensorflow.gfile as gfile
import h5py
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import sys


def RANK(x):
    '''cross-section rank in each day'''
    ret = np.zeros(x.shape)*np.nan
    for i in range(x.shape[0]):
        temp = x[i,:]
        idx = np.where( (np.isnan(temp) or np.isinf(temp)) == False)[0]
        temp = temp[idx]
        temp1 = np.zeros(temp.shape)
        temp0 = np.int32(np.argsort(temp))
        temp1[temp0] = range(len(temp0))
        ret[i,idx]=temp1
    return ret


# def evaluation(test_year, train_period, step2_Dataset = 2, chosen_stocks=500, \
#                   start_cash = 1000000,   service_charge_rate = 0.0015, \
#                   step2_slice_range = 60, step2_slice_stride = 1, 
#                   step2_pred_period = 1, percentage = 0.1, stratified_num = 3, \
#                   decay_period = 3):

test_year=2017
train_period=3
step2_Dataset = 6
step2_slice_range = 60
step2_slice_stride = 1
step2_pred_period = 1
chosen_stocks = 500
percentage = 0.1
stratified_num = 3
start_cash = 1000000
service_charge_rate = 0.003
decay_period = 10


# 存、读路径
step4_model_path = "./history"+'/Dataset'+str(step2_Dataset)+'-stock_number'+str(chosen_stocks)+'-TestYear'+\
                    str(test_year)+'-TrainPeriod'+str(train_period) +'-'+\
                    'SliceRange'+str(step2_slice_range) + '-stride'+\
                    str(step2_slice_stride) + '-pred'+str(step2_pred_period)
step4_read_dir_1 = '../../../dataset/training_data/'+'Dataset'+str(step2_Dataset)+'-stock_number'+str(chosen_stocks)+\
                    '-TestYear'+str(test_year)+'-TrainPeriod'+str(train_period) +\
                    '-'+'SliceRange'+str(step2_slice_range) + '-stride'+\
                    str(step2_slice_stride) + '-pred'+str(step2_pred_period)
step4_read_dir_2 = '../../../dataset/raw_data'
step4_save_dir = 'evaluation_plot'+'/Dataset'+str(step2_Dataset)+'-stock_number'+str(chosen_stocks)+'-TestYear'+\
                  str(test_year)+'-TrainPeriod'+str(train_period) +'-'+'SliceRange'+\
                  str(step2_slice_range) + '-stride'+str(step2_slice_stride) + \
                  '-pred'+str(step2_pred_period)



# 文件读取
original_time_column = pd.read_csv('../../../dataset/raw_data/CLOSE.csv')['time'].values
# df_temp_file = pd.read_csv('../../../dataset/raw_data/CLOSE.csv', parse_dates=['time'])
print('Reading data...')
### 加载数据
f = np.load(step4_read_dir_1 + '/test.npz')
x_test, y_test = f['x_test'], f['y_test']
time_recorder, stock_recorder = f['time_recorder'], f['stock_recorder']
f.close()

f = np.load(step4_read_dir_2 + '/RET.npz')
RET = f['RET'][...,0:chosen_stocks]
f.close()



# 整理数据格式
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# 保存结果的文件夹
if gfile.Exists(step4_save_dir)==False:
    gfile.MakeDirs(step4_save_dir)


### 加载模型
print('Loding model...')

model = load_model(step4_model_path)
print('The model imported is '+ step4_model_path)

##########################################
############# 开始测试模型效果###########
##########################################
predicted_yield_rate = model.predict(x_test, verbose=1)


# 将time_recorder 转换为时间戳比大小
time_recorder_time_stamp = []
for i in range(len(time_recorder)):
    time_recorder_time_stamp.append(time.mktime(datetime.datetime.strptime(time_recorder[i], "%Y/%m/%d").timetuple()))


time_recorder_time_stamp = np.array(time_recorder_time_stamp)

time_sorted_idx = np.argsort(time_recorder_time_stamp)
time_recorder = time_recorder[time_sorted_idx]
stock_recorder = stock_recorder[time_sorted_idx]
predicted_yield_rate = predicted_yield_rate[time_sorted_idx]
y_test = y_test[time_sorted_idx]
time_recorder_time_stamp = time_recorder_time_stamp[time_sorted_idx]


# temp_uniq_time_recorder = np.unique(time_recorder)

# num_recorder用于记录每日可用的测试样本的个数
unique_time_recorder = np.unique(time_recorder_time_stamp).copy()
num_recorder = np.zeros(len(unique_time_recorder))
for i in range(len(unique_time_recorder)):
    num_recorder[i] = np.sum(time_recorder_time_stamp == unique_time_recorder[i])


num_recorder = num_recorder.astype(np.int64)


pred_yield_rate_per_day = -np.inf*np.ones([len(unique_time_recorder), chosen_stocks])
real_yield_rate_per_day = -np.inf*np.ones([len(unique_time_recorder), chosen_stocks])
stock_recorder_per_day = -np.inf*np.ones([len(unique_time_recorder), chosen_stocks])
two_d_pred_yield_rate = np.nan*np.ones([len(unique_time_recorder), chosen_stocks])
two_d_real_yield_rate = np.nan*np.ones([len(unique_time_recorder), chosen_stocks])
 
# 记录换手率
turnover_rate = np.nan*np.zeros([len(unique_time_recorder), stratified_num])
# 记录每日筛选后的股票预测收益率、真实收益率和代码
for i in range(len(unique_time_recorder)):
    for j in range(chosen_stocks):
        if j < num_recorder[i]:
            pred_yield_rate_per_day[i,j] = predicted_yield_rate[np.sum(num_recorder[0:i])+j]
            real_yield_rate_per_day[i,j] = y_test[np.sum(num_recorder[0:i])+j]
            stock_recorder_per_day[i,j] = stock_recorder[np.sum(num_recorder[0:i])+j]
        else:
            break


for i in range(len(unique_time_recorder)):
    two_d_pred_yield_rate[i, stock_recorder_per_day[i,0:num_recorder[i]].astype(int)] =\
                                             pred_yield_rate_per_day[i,0:num_recorder[i]]
    two_d_real_yield_rate[i, stock_recorder_per_day[i,0:num_recorder[i]].astype(int)] =\
                                             real_yield_rate_per_day[i,0:num_recorder[i]]





# 利用decay方法对收益率排序（按照decay_period天的rank平均值）
pred_ret = np.zeros(two_d_pred_yield_rate.shape)*(np.nan)
real_ret = np.zeros(two_d_real_yield_rate.shape)*(np.nan)
pred_ret[0:decay_period-1,...] = two_d_pred_yield_rate[0:decay_period-1,...]
real_ret[0:decay_period-1,...] = two_d_real_yield_rate[0:decay_period-1,...]


for j in range(two_d_pred_yield_rate.shape[1]):
    temp = two_d_pred_yield_rate[:,j]
    idx = np.where(np.isnan(temp) == False)[0]
    temp = temp[idx]
    for i in range(decay_period-1,len(idx)):
        pred_ret[idx[i],j] = np.mean(temp[i-decay_period+1:i+1])


for j in range(two_d_real_yield_rate.shape[1]):
    temp = two_d_real_yield_rate[:,j]
    idx = np.where(np.isnan(temp) == False)[0]
    temp = temp[idx]
    for i in range(decay_period-1,len(idx)):
        real_ret[idx[i],j] = np.mean(temp[i-decay_period+1:i+1])



pred_ret[np.where(np.isnan(pred_ret))] = -np.inf
real_ret[np.where(np.isnan(real_ret))] = -np.inf



# 每行降序排序
sorted_yield_rate_per_day = np.sort(pred_ret, axis=1)[ :,  : :  -1]
# 寻找 sorted_yield_rate_per_day_index 对应的真实收益率，和股票相对位置
sorted_real_yield_rate_per_day = np.nan*np.zeros([pred_yield_rate_per_day.shape[0] , pred_yield_rate_per_day.shape[1]])
sorted_stock_recorder_per_day = np.nan*np.zeros([pred_yield_rate_per_day.shape[0] , pred_yield_rate_per_day.shape[1]])


# 记录排列后的顺序
sorted_stock_recorder_per_day = np.argsort(pred_ret, axis=1)[ :, : : -1]


for i in range(sorted_real_yield_rate_per_day.shape[0]):
    sorted_real_yield_rate_per_day[i,:] = real_ret[i , sorted_stock_recorder_per_day[i,:]]
    


# 将没有数据的位置替换为nan
sorted_yield_rate_per_day[np.where(np.isinf(sorted_yield_rate_per_day))] = np.nan
sorted_real_yield_rate_per_day[np.where(np.isinf(sorted_real_yield_rate_per_day))] = np.nan
# sorted_stock_recorder_per_day[np.where(np.isinf(sorted_stock_recorder_per_day))] = np.nan

sorted_stock_recorder_per_day=sorted_stock_recorder_per_day.astype(float)
sorted_stock_recorder_per_day[np.isnan(sorted_yield_rate_per_day)] = np.nan






### 组间收益对比
print('Plotting the yield rate of stratified portfolio of each day...')
# cash记录不同分组下的仓位
cash = np.zeros([pred_yield_rate_per_day.shape[0] , pred_yield_rate_per_day.shape[1] , stratified_num])

# 第一天的仓位（平均仓位）
for temp in range(stratified_num):
    stock_num_in_group = np.floor(num_recorder[0]*percentage)
    stock_num_in_stratification = np.floor(num_recorder[0]/stratified_num)
    if temp+1 < (stratified_num+1)/2.0:
        temp_rate = sorted_real_yield_rate_per_day[\
                            0, int(temp*stock_num_in_stratification) : \
                            int(temp*stock_num_in_stratification+stock_num_in_group)]
        temp_stock_recorder = sorted_stock_recorder_per_day[\
                            0, int(temp*stock_num_in_stratification) : \
                            int(temp*stock_num_in_stratification+stock_num_in_group)]
        # 建仓
        cash[0, temp_stock_recorder.astype(int), temp] = (start_cash/len(temp_stock_recorder))*np.ones(len(temp_stock_recorder))
        # 扣除手续费
        cash[0, temp_stock_recorder.astype(int), temp] = cash[0, temp_stock_recorder.astype(int), temp]*(1 - service_charge_rate)
        # 加上收益
        cash[0, temp_stock_recorder.astype(int), temp] = cash[0, temp_stock_recorder.astype(int), temp]*(1+temp_rate)
    elif temp+1 > (stratified_num+1)/2.0:
        temp_rate = (sorted_real_yield_rate_per_day[\
                            0, int((temp+1)*stock_num_in_stratification-stock_num_in_group) : \
                            int((temp+1)*stock_num_in_stratification)])
        temp_stock_recorder = (sorted_stock_recorder_per_day[\
                            0, int((temp+1)*stock_num_in_stratification-stock_num_in_group) : \
                            int((temp+1)*stock_num_in_stratification)])
        # 建仓
        cash[0, temp_stock_recorder.astype(int), temp] = (start_cash/len(temp_stock_recorder))*np.ones(len(temp_stock_recorder))
        # 扣除手续费
        cash[0, temp_stock_recorder.astype(int), temp] = cash[0, temp_stock_recorder.astype(int), temp]*(1 - service_charge_rate)
        # 加上收益
        cash[0, temp_stock_recorder.astype(int), temp] = cash[0, temp_stock_recorder.astype(int), temp]*(1+temp_rate)
    else:
        temp_rate = (sorted_real_yield_rate_per_day[\
                            0, int( np.floor(num_recorder[0]/2-stock_num_in_group/2)) : \
                            int(np.floor(num_recorder[0]/2-stock_num_in_group/2)+stock_num_in_group)])
        temp_stock_recorder = (sorted_stock_recorder_per_day[\
                            0, int( np.floor(num_recorder[0]/2-stock_num_in_group/2)) : \
                            int(np.floor(num_recorder[0]/2-stock_num_in_group/2)+stock_num_in_group)])
        # 建仓
        cash[0, temp_stock_recorder.astype(int), temp] = (start_cash/len(temp_stock_recorder))*np.ones(len(temp_stock_recorder))
        # 扣除手续费
        cash[0, temp_stock_recorder.astype(int), temp] = cash[0, temp_stock_recorder.astype(int), temp]*(1 - service_charge_rate)
        # 加上收益
        cash[0, temp_stock_recorder.astype(int), temp] = cash[0, temp_stock_recorder.astype(int), temp]*(1+temp_rate)



# 第二天开始至期末的收益就可以开始迭代。并记录换手率
for day in range(1,len(num_recorder)):
    for temp in range(stratified_num):
        stock_num_in_group = np.floor(num_recorder[day]*percentage)
        stock_num_in_stratification = np.floor(num_recorder[day]/stratified_num)
        if temp+1 < (stratified_num+1)/2.0:
            temp_rate = sorted_real_yield_rate_per_day[\
                                day, int(temp*stock_num_in_stratification) : \
                                int(temp*stock_num_in_stratification+stock_num_in_group)]
            temp_stock_recorder = sorted_stock_recorder_per_day[\
                                day, int(temp*stock_num_in_stratification) : \
                                int(temp*stock_num_in_stratification+stock_num_in_group)]
            # 按照模型，不扣除手续费的建仓状态
            cash[day, temp_stock_recorder.astype(int), temp] = (np.nansum(cash[day-1, \
                                    ..., temp])/len(temp_stock_recorder))*np.ones(len(temp_stock_recorder))
            # 仓位变动绝对值和
            turnover = np.sum(np.abs(cash[day,..., temp] - cash[day-1,..., temp]))
            # 换手率
            turnover_rate[day, temp] = turnover/np.nansum(cash[day,..., temp])
            # 手续费总额：
            fee = turnover*service_charge_rate
            # 扣除手续费
            cash[day, temp_stock_recorder.astype(int), temp] = cash[day, temp_stock_recorder.astype(int), temp] - fee/len(temp_stock_recorder)
            # 加上收益
            cash[day, temp_stock_recorder.astype(int), temp] = cash[day, temp_stock_recorder.astype(int), temp]*(1+temp_rate)
        elif temp+1 > (stratified_num+1)/2.0:
            temp_rate = (sorted_real_yield_rate_per_day[\
                                day, int((temp+1)*stock_num_in_stratification-stock_num_in_group) : \
                                int((temp+1)*stock_num_in_stratification)])
            temp_stock_recorder = (sorted_stock_recorder_per_day[\
                                day, int((temp+1)*stock_num_in_stratification-stock_num_in_group) : \
                                int((temp+1)*stock_num_in_stratification)])
            # 按照模型，不扣除手续费的建仓状态
            cash[day, temp_stock_recorder.astype(int), temp] = (np.nansum(cash[day-1, \
                                    ..., temp])/len(temp_stock_recorder))*np.ones(len(temp_stock_recorder))
            # 仓位变动绝对值和
            turnover = np.sum(np.abs(cash[day,..., temp] - cash[day-1,..., temp]))
            # 换手率
            turnover_rate[day, temp] = turnover/np.nansum(cash[day,..., temp])
            # 手续费总额：
            fee = turnover*service_charge_rate
            # 扣除手续费
            cash[day, temp_stock_recorder.astype(int), temp] = cash[day, temp_stock_recorder.astype(int), temp] - fee/len(temp_stock_recorder)
            # 加上收益
            cash[day, temp_stock_recorder.astype(int), temp] = cash[day, temp_stock_recorder.astype(int), temp]*(1+temp_rate)
        else:
            temp_rate = (sorted_real_yield_rate_per_day[\
                                day, int( np.floor(num_recorder[day]/2-stock_num_in_group/2)) : \
                                int(np.floor(num_recorder[day]/2-stock_num_in_group/2)+stock_num_in_group)])
            temp_stock_recorder = (sorted_stock_recorder_per_day[\
                                day, int( np.floor(num_recorder[day]/2-stock_num_in_group/2)) : \
                                int(np.floor(num_recorder[day]/2-stock_num_in_group/2)+stock_num_in_group)])
            # 按照模型，不扣除手续费的建仓状态
            cash[day, temp_stock_recorder.astype(int), temp] = (np.nansum(cash[day-1, \
                                    ..., temp])/len(temp_stock_recorder))*np.ones(len(temp_stock_recorder))
            # 仓位变动绝对值和
            turnover = np.sum(np.abs(cash[day,..., temp] - cash[day-1,..., temp]))
            # 换手率
            turnover_rate[day, temp] = turnover/np.nansum(cash[day,..., temp])
            # 手续费总额：
            fee = turnover*service_charge_rate
            # 扣除手续费
            cash[day, temp_stock_recorder.astype(int), temp] = cash[day, temp_stock_recorder.astype(int), temp] - fee/len(temp_stock_recorder)
            # 加上收益
            cash[day, temp_stock_recorder.astype(int), temp] = cash[day, temp_stock_recorder.astype(int), temp]*(1+temp_rate)



# 输出换手率信息
mean_turnover_rate = np.nanmean(turnover_rate, axis=0)
print("\033[1;31mturnover rate of each group: "+ str(mean_turnover_rate) + '\033[0m')
# 将cash转化为收益率
temp_cum_yield_rate = np.nan*np.zeros([cash.shape[0]+1, stratified_num])
cum_yield_rate = np.nan*np.zeros([cash.shape[0], stratified_num])
temp_cum_yield_rate[0,:] = start_cash
temp_cum_yield_rate[1:,:] = np.nansum(cash,axis=1)

for i in range(1,temp_cum_yield_rate.shape[0]):
    cum_yield_rate[i-1,:] = temp_cum_yield_rate[i,:]/temp_cum_yield_rate[i-1,:] - 1



fig1 = plt.figure()
for temp in range(stratified_num):
    yield_data = np.nancumprod(cum_yield_rate[:,temp]+1)-1
    plt.plot(np.arange(len(num_recorder)) ,  yield_data  , '-',
        label = '#'+str(temp)+': '+str(100*yield_data[-1])+"%")


# 大盘
mean_yield_rate_per_day = np.nanmean(sorted_real_yield_rate_per_day, axis=1)
yield_data = np.cumprod(mean_yield_rate_per_day+1)-1
plt.plot(np.arange(len(mean_yield_rate_per_day)) , \
                     yield_data,\
                    'r-.',
                    label = 'Market: '+str(100*yield_data[-1])+"%" )





ax = plt.gca()
ax.set_title( str(test_year)+' selected yield rate.')  
ax.set_xlabel("Time")
ax.set_ylabel("Yield Rate") 
plt.legend(loc='upper left')
plt.savefig(step4_save_dir+'/yield_rate_curve.png', dpi=fig1.dpi)


# OLS Line
Y = np.nancumprod(sorted_real_yield_rate_per_day+1, axis=0)[-1,:]-1
Y = Y[0:min(num_recorder)]
X = np.arange(len(Y))
X = sm.add_constant(X)

ols_model = sm.OLS(Y,X)
ols_model_result = ols_model.fit()


fig2 = plt.figure()
plt.scatter(np.arange(len(Y)),Y,c='blue')
k = ols_model_result.params[1]
b = ols_model_result.params[0]
plt.plot(np.arange(len(Y)), k*np.arange(len(Y))+b, c='r',linewidth=3)
# 全年chosen_stocks所有股票的收益
total = np.nancumprod(np.nanmean(sorted_real_yield_rate_per_day+1,axis=1))[-1]-1
plt.plot(np.arange(len(Y)), total*np.ones(len(Y)), c='black',linestyle='dashed',linewidth=2)


ax = plt.gca()
ax.set_title( 'Sorted stocks and OLS')  
ax.set_xlabel("Sorted number")
ax.set_ylabel("Cummulative Yield Rate") 
fig2.savefig(step4_save_dir+'/sorted_pred_yield_rate.png', dpi=fig2.dpi)

# yield rate data
np.savetxt(step4_save_dir+"/sorted_pred_yield_rate.csv", sorted_yield_rate_per_day, delimiter=',')
np.savetxt(step4_save_dir+"/sorted_real_yield_rate.csv", sorted_real_yield_rate_per_day, delimiter=',')

