import numpy as np
import os
import pandas as pd

chosen_stocks = 500
step2_Dataset = np.array([1,2,3,4,6])
step2_slice_range = 60
step2_slice_stride = 1
step2_pred_period = 1
test_year_set = np.arange(2008,2019,1)
train_period_set = np.array([3])
percentage = 0.1
stratified_num = 3

# 保存各dataset、各年的3组分层结果
result_recorder = np.nan*np.zeros([len(step2_Dataset), len(test_year_set), stratified_num])
# 保存对应的均值、方差
stat_recorder = np.nan*np.zeros([len(step2_Dataset), 2])


dataset_idx = -1
### 模型训练和评估测试
for dataset in step2_Dataset:
    dataset_idx += 1
    test_year_idx = -1
    print("\033[1;31;43m"+"dataset:"+str(dataset)+ '\033[0m')  
    for test_year in test_year_set:
        test_year_idx += 1
        print("\033[1;31;43m"+"test year:"+str(test_year)+ '\033[0m')
        for train_period in train_period_set:
            print("\033[1;31;43m"+"train period:"+str(train_period)+ '\033[0m')
            step4_save_dir = 'evaluation_plot'+'/Dataset'+str(dataset)+'-stock_number'+str(chosen_stocks)+'-TestYear'+\
                      str(test_year)+'-TrainPeriod'+str(train_period) +'-'+'SliceRange'+\
                      str(step2_slice_range) + '-stride'+str(step2_slice_stride) + \
                      '-pred'+str(step2_pred_period)
            sorted_real_yield_rate_per_day = pd.read_csv(step4_save_dir + '/sorted_real_yield_rate.csv', header = -1).values
            num_recorder = sorted_real_yield_rate_per_day.shape[1] - np.sum(np.isnan(sorted_real_yield_rate_per_day),1)
            cum_yield_rate = np.ones([len(num_recorder), stratified_num])
            for day in range(len(num_recorder)):
                for temp in range(stratified_num):
                    stock_num_in_group = np.floor(num_recorder[day]*percentage)
                    stock_num_in_stratification = np.floor(num_recorder[day]/stratified_num)
                    if temp+1 < (stratified_num+1)/2.0:
                        cum_yield_rate[day, temp] = np.nanmean(sorted_real_yield_rate_per_day[day, int(temp*stock_num_in_stratification) : int(temp*stock_num_in_stratification+stock_num_in_group)])
                    elif temp+1 > (stratified_num+1)/2.0:
                        cum_yield_rate[day, temp] = np.nanmean(sorted_real_yield_rate_per_day[day, int((temp+1)*stock_num_in_stratification-stock_num_in_group) : int((temp+1)*stock_num_in_stratification)])
                    else:
                        cum_yield_rate[day, temp] = np.nanmean(sorted_real_yield_rate_per_day[day, int( np.floor(num_recorder[day]/2-stock_num_in_group/2))    :int(np.floor(num_recorder[day]/2-stock_num_in_group/2)+stock_num_in_group)])
            yield_data = np.nancumprod(cum_yield_rate+1,0)-1
            result_recorder[dataset_idx, test_year_idx, : ] = yield_data[-1,:]


for dataset_idx in range(len(step2_Dataset)):
    temp = result_recorder[dataset_idx, :,  0]-result_recorder[dataset_idx, :, -1]
    stat_recorder[dataset_idx, 0] = np.mean(temp)
    stat_recorder[dataset_idx, 1] = np.std(temp, ddof=1)
    print("The mean value of dataset"+str(step2_Dataset[dataset_idx])+" is "+str((100*np.mean(temp)).round(5))+\
            "%, the std is "+str(np.std(temp, ddof=1).round(4))+", the max is "+str(np.max(temp).round(4))+" in "+str(test_year_set[np.where(temp==np.max(temp))])+\
            ", the min is "+str(np.min(temp).round(4))+" in " +str(test_year_set[np.where(temp==np.min(temp))]))



        