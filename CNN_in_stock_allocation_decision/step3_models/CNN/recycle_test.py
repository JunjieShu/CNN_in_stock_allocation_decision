import numpy as np
import os
from model_train import *
from evaluation_decay_and_fee import *

stock_num = 500
step2_Dataset = np.array([6])
step2_slice_range = 60
step2_slice_stride = 1
step2_pred_period = 5
test_year_set = np.arange(2013,2019,1)
train_period_set = np.array([3])


### 模型训练和评估测试
for dataset in step2_Dataset:
    print("\033[1;31;43m"+"dataset:"+str(dataset)+ '\033[0m')  
    for test_year in test_year_set:
        print("\033[1;31;43m"+"test year:"+str(test_year)+ '\033[0m')
        for train_period in train_period_set:
            print("\033[1;31;43m"+"train period:"+str(train_period)+ '\033[0m')

            data_dir = '../../../dataset/training_data/'+'Dataset'+str(dataset)+'-stock_number'+str(stock_num)+\
                        '-TestYear'+str(test_year)+'-TrainPeriod'+str(train_period) +\
                        '-'+'SliceRange'+str(step2_slice_range) + '-stride'+\
                        str(step2_slice_stride) + '-pred'+str(step2_pred_period)
            model_dir = './history/Dataset'+str(dataset)+'-stock_number'+str(stock_num)+\
                        '-TestYear'+str(test_year)+'-TrainPeriod'+str(train_period) +\
                        '-'+'SliceRange'+str(step2_slice_range) + '-stride'+\
                        str(step2_slice_stride) + '-pred'+str(step2_pred_period)

            if os.path.exists(data_dir) == False:
                print('\n')
                continue

            if os.path.exists(model_dir) == False:
                print('Training...')
                model_train(test_year=test_year, train_period=train_period, Continue_model=False, epoch=3, step2_Dataset = dataset, 
                  step2_slice_range = 60, step2_slice_stride = 1, chosen_stocks = 500,
                  step2_pred_period = 5,  evaluate_test = False )
                evaluation(test_year=test_year, train_period=train_period, step2_Dataset = dataset, chosen_stocks=500,
                            step2_slice_range = 60, step2_slice_stride = 1, 
                            step2_pred_period = 5, percentage = 0.1, stratified_num = 3)
            else:
                evaluation(test_year=test_year, train_period=train_period, step2_Dataset = dataset, chosen_stocks=500,
                            step2_slice_range = 60, step2_slice_stride = 1, 
                            step2_pred_period = 5, percentage = 0.1, stratified_num = 3)






