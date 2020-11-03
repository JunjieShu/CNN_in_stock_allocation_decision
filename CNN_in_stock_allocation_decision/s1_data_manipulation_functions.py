import os
import numpy as np
import pandas as pd
import sys
import time
import datetime
import progressbar



# step1_factors = ['AMOUNT', 'CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME']

# step1_factors = ['perc_AMOUNT', 'perc_CLOSE', 'perc_HIGH', 'perc_LOW',\
#                  'perc_OPEN', 'perc_VOLUME']

# step1_factors = ['perc_AMOUNT', 'perc_CLOSE', 'perc_HIGH', 'perc_LOW',\
#                  'perc_OPEN', 'perc_VOLUME', \
#                  'MOM_1', 'MOM_2', 'MOM_3']

# step1_factors = ['AMOUNT', 'CLOSE', 'HIGH', 'LOW',\
#                  'OPEN', 'VOLUME',  'EMA_10','EMA_20','EMA_50',\
#                   'MOM_1', 'MOM_2', 'MOM_3']


step1_factors = ['perc_AMOUNT', 'perc_CLOSE', 'perc_HIGH', 'perc_LOW',\
                 'perc_OPEN', 'perc_VOLUME', \
                 'MOM_1', 'MOM_2', 'MOM_3',\
                 'perc_SMA_5', 'perc_SMA_10','perc_SMA_30']

def step1_reshape_data(step1_factors):
    '''
    This function read factors and RET data, and reshape them as:
        the first dimension is factor
        the second dimension is time
        the third dimension is matrix wrt category of stocks

        after saving them, we'll use it in step2
    这个函数的作用是将已经处理好的factors和RET数据读取，然后将它们reshape为第一维度为因子、
    第二维度为时间、第三维度为股票种类的矩阵。保存后将在step2函数中使用
    '''
    print('Beginning at: ',time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())))

    # 设置读取和保存文件夹
    step1_read_dir = './../dataset/factors/read_factors'
    step1_read_RET_dir = './../dataset/raw_data'
    step1_save_dir = './../dataset/step1_generated_reshaped_factors_and_labels'

    ################文件读取部分 file reading################
    ###  header and time column 表头和时间列
    header = pd.read_csv('./../dataset/raw_data/CLOSE.csv', header=None).values[0,:][1:]
    time_column = pd.read_csv('./../dataset/raw_data/CLOSE.csv')['time'].values

    ### read factors
    print('Reading the factors selected in step0 / manually...')
    data = np.zeros([len(time_column), len(header), len(step1_factors)])*np.nan 
    count = 0
    for factor in step1_factors:
        data[..., count] = pd.read_csv(step1_read_dir+'/'+factor+'.csv', header=-1).values
        count = count + 1
        print('    factor '+factor+' is imported successfully...' )

    ### read RET
    f = np.load(step1_read_RET_dir+"/RET.npz")
    RET = f['RET']
    f.close()
    print('    file RET.npz is imported successfully...' )




    ################reshape################
    # reshape 后的 shape 跟研报中保持一致
    # 一开始data的shape为(len(time_column), len(header), len(step1_factors))
    # 因此只需要“将第三个shape换到第一个shape上”
    temp_data = np.zeros([len(step1_factors), len(time_column), len(header)])*np.nan
    for count in range(len(step1_factors)):
        temp_data[count,...] = data[..., count].copy()

    data = temp_data.copy()
    del temp_data


    temp_RET = np.zeros([1, len(time_column), len(header)])*np.nan
    temp_RET[0,...] = RET.copy()
    RET = temp_RET.copy()
    del temp_RET


    ################输出部分################
    print('Generating the npz files...')
    # 创建文件保存路径
    if os.path.exists(step1_save_dir) == False:
        os.makedirs(step1_save_dir)

    # 计算已有文件夹个数，并创建文件夹
    count = 0
    for dir in os.listdir(step1_save_dir):
        if os.path.isdir(step1_save_dir+'/'+dir):
            count += 1

    while True:
        new_folder = step1_save_dir+'/Dataset'+str(count+1)
        if os.path.exists(new_folder) == False:
            print('Creating folder "Dataset'+str(count+1)+'"')
            os.makedirs(new_folder)
            break
        else:
            count += 1

    # 将读取的因子名写入new_folder中的一个txt中
    print('Creating a new txt recording the factor names...')
    file = open(new_folder+'/selected_factor_names.txt','w')

    count = 0
    for factor in step1_factors:
            count = count + 1
            file.write(str(count)+'. '+factor+' \n')
    file.close()

    # 保存factors和labels
    np.savez(new_folder+"/factors.npz", factors = data.copy())
    np.savez(new_folder+"/labels.npz", labels = RET.copy())

    print('Ending at: ',time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())))






def step2_train_and_test_data(test_year=2008,   train_period=3,  step2_Dataset = 6, 
                                step2_slice_range = 60, step2_slice_stride = 5, 
                                step2_pred_period = 5, chosen_stocks=500):

    '''
    参数说明:
    test_year: 测试年份
    train_period: 训练周期（年)
    step2_Dataset: 选择的已填充数据集
    step2_slice_range: 每个样本的时间序列长度
    step2_slice_stride: 样本的切片步长
    step2_pred_period: 收益率累乘天数（当样本的切片步长不为 1 的时候，那么预测的收益率将会是在切片步长的时间周期长度类的累计收益率）
    chosen_stock: 选取训练的股票支数
    '''
    # test_year=2008
    # train_period=3
    # step2_Dataset = 6
    # step2_slice_range = 60
    # step2_slice_stride = 5
    # step2_pred_period = 5
    # chosen_stocks = 500

    print('Beginning at: ',time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())))

    # 设置相关路径
    step2_read_dir = './../dataset/step1_generated_reshaped_factors_and_labels' +'/Dataset'+str(step2_Dataset)
    step2_save_dir = './../dataset/training_data/'+'Dataset'+str(step2_Dataset)+'-stock_number'+str(chosen_stocks)+'-TestYear'+str(test_year)+'-TrainPeriod'+str(train_period) +'-'+'SliceRange'+str(step2_slice_range) + '-stride'+str(step2_slice_stride) + '-pred'+str(step2_pred_period)

    # 创建保存文件夹
    print('Creating saving folder...')
    if os.path.exists(step2_save_dir) == False:
        os.makedirs(step2_save_dir)


    ############读取数据##############
    print('Importing data...')
    # 神经网络labels数据:
    f = np.load(step2_read_dir+"/labels.npz")
    RET = f['labels'][:,:,0:chosen_stocks]
    f.close()
    print('    Labels(RET) is imported successfully...' )
    # 神经网络factors数据:
    f = np.load(step2_read_dir+"/factors.npz")
    factors = f['factors'][:,:,0:chosen_stocks]
    f.close()
    print('    factors is imported successfully...' )

    # temp1=np.where(np.isnan(factors[0,...]))
    # len(temp1[0])
    # temp2=np.where(np.isnan(factors[5,...]))
    # len(temp2[0])

    original_time_column = pd.read_csv('./../dataset/raw_data/CLOSE.csv')['time'].values

    CLOSE = pd.read_csv('./../dataset/raw_data/CLOSE.csv',header=-1).values[1:,1:].astype(np.float64)[:,0:chosen_stocks]
    stock_code = pd.read_csv('./../dataset/raw_data/CLOSE.csv',header=-1).values[0,1:][0:chosen_stocks]
    factor_number = factors.shape[0]
    stock_number = factors.shape[2]


    ### 记录股票上市时间的index
    tradeday = pd.read_csv('./../dataset/raw_data/tradeday.csv',header=-1).values



    TTM = pd.read_csv('./../dataset/raw_data/TTM.csv',header=-1).values
    for i in range(TTM.shape[0]):
        TTM[i,0] = "SZ#"+TTM[i,0][0:6]
        TTM[i,1] = TTM[i,1][0:-1]


    TTM_distance = np.nan*np.zeros([stock_number])
    first_day_idx = np.where(original_time_column[0]==tradeday)[0][0]

    for i in range(stock_number):
        time_temp_idx = np.where(TTM[np.where(TTM[:,0]==stock_code[i]),1][0] == tradeday)[0]
        if len(time_temp_idx)==0:
            TTM_distance[i] = -np.inf
        else:
            TTM_distance[i] = time_temp_idx - first_day_idx
        

    ### 去掉上市后的60天数据
    for i in range(len(TTM_distance)):
        # 在有数据的第一天的60天以前已经上市:
        if TTM_distance[i] + 60 <= 0:
            continue
        # 在有数据的最后一天之后仍然未上市:
        elif TTM_distance[i] > RET.shape[1]:
            continue
        # 在有数据的最后一天之前60天内上市:
        elif TTM_distance[i] + 60 - 1 > RET.shape[1]:
            RET[..., i] = np.nan    # RET需要比factor 少一天
            factors[..., i] = np.nan
            CLOSE[...,i] = np.nan
        # 其他情况:
        else: 
            RET[..., 0:np.int(TTM_distance[i]+60+1), i] = np.nan    # RET需要比factor 少一天
            factors[..., 0:np.int(TTM_distance[i]+60), i] = np.nan
            CLOSE[0:np.int(TTM_distance[i]+60), i] = np.nan


    ### 去掉收益率绝对值大于20%后的60天 
    for i in range(RET.shape[1]):
        for j in range(stock_number):
            if np.isnan(RET[0,i,j]):
                continue
            elif np.abs(RET[0,i,j]) >= 0.2:
                if i + 60 - 1 < RET.shape[1]:
                    RET[0,i:i+60+1,j] = np.nan
                    factors[:,i:i+60,j] = np.nan
                    CLOSE[i:i+60,j] = np.nan
                else:
                    RET[0,i:,j] = np.nan
                    factors[:,i:,j] = np.nan
                    CLOSE[i:,j] = np.nan


    ### 去掉CLOSE出现负数的前后30天
    for i in range(CLOSE.shape[1]):
        for j in range(stock_number):
            if CLOSE[i,j] > 0 or np.isnan(CLOSE[i,j]):
                continue
            else:
                begin_idx = np.int(((np.sign(i-30)+1)*(i-30))/2)
                end_idx = np.int((i+30)-((np.sign(i+30-factors.shape[1])+1)*(i+30-factors.shape[1]))/2)
                RET[0,begin_idx:end_idx,j] = np.nan
                factors[:,begin_idx:end_idx,j] = np.nan
                CLOSE[begin_idx:end_idx,j] = np.nan




    #######################################
    ################# 切片 ################
    #######################################
    print('Slicing the factors and labels')

    ####################################
    #############训练数据切片#############
    ####################################
    train_year = np.arange(test_year - train_period, test_year)
    print('    Train year is '+str(train_year)+'...')
    original_train_year_idx = np.zeros([len(original_time_column)])
    for count_time_column in range(len(original_time_column)):
        if int(original_time_column[count_time_column][0:4]) in train_year:
            original_train_year_idx[count_time_column] = 1


    # 首先判断一下每个因子的nan比例
    total_size = factors.shape[1]*factors.shape[2]
    CLOSE_nan_percentage = np.sum(np.isnan(CLOSE))/total_size
    nan_percentage_of_each_factors = np.nan*np.zeros(factors.shape[0])
    for factor_num_count in range(factors.shape[0]):
        nan_percentage_of_each_factors[factor_num_count] = \
            np.sum(np.isnan(factors[factor_num_count,...]))/total_size

    # 对于nan比例大于0.05的因子，删掉
    # 如果没有nan大于临界值的因子，那么只需要删除未上市/停牌后的值，剩下的填充
    del_factors_idx = np.where((nan_percentage_of_each_factors-CLOSE_nan_percentage)>0.05)[0]
    if del_factors_idx.shape[0] != 0:
        print("\033[1;31mThere are factors NaN percentage greater than 0.05 reletive CLOSE" + '\033[0m')
        print("\033[1;31mTheir index are:" + str(del_factors_idx) + '\033[0m')
        print("\033[1;31mDeleting..." + '\033[0m')
        factors = np.delete(factors, del_factors_idx, axis=0)
    else:
        pass
       

    # //是向下取整除法
    # sample_num大于最大的样本数量，在有nan时可能填不满，因此有一些可能需要删除
    # 这里对nan的处理方法还是采用因子函数那样的方法。
    print('    Preparing training samples...')
    sample_num = stock_number*((np.sum(original_train_year_idx==1)) // step2_slice_stride + 1)
    x_train = np.nan*np.zeros([sample_num, factor_number, step2_slice_range])
    y_train = np.nan*np.zeros(sample_num)
    count = 0 # 用count对x_train和y_train计数




    # original_train_begin_date_idx是original_time_column的索引。original_time_column是没有
    # 经过nan修正的，因此可以作为所有股票的统一日程规范。经过每一次循环后，
    # original_train_begin_date_idx就加上step2_slice_stride, 并且在前step2_pred_period的索引范围内
    # 找某只股票有没有对应的日期。有的话就定下这几天内的累计收益率，并且
    # 切片original_train_begin_date_idx - step2_pred_period 前step2_slice_range天（经过nan）修正后
    # 的数据为x

    # 所有的xxx_date_idx都将是样本的累计收益率的最后一天的索引
    # 首先通过original_time_column定位每一次切片的日期。这样可以保证在step2_slice_stride ！= 1
    # 时，切片不会由于nan的出现而错位。这样对之后evaluation部分再换仓部分的计算比较便利
    original_train_begin_date_idx_set = np.zeros(original_time_column.shape)
    for original_train_begin_date_idx in range(len(original_time_column)):
        if int(original_time_column[original_train_begin_date_idx][0:4]) in train_year:
            original_train_begin_date_idx_set[original_train_begin_date_idx] = 1


    train_begin_date_idx = np.min(np.where(original_train_begin_date_idx_set==1)[0])
    train_end_date_idx = np.max(np.where(original_train_begin_date_idx_set==1)[0])+1



    # 确定original中的索引起始点:
    original_date_idx = train_begin_date_idx + step2_pred_period  

    # 判断当前的original_train_begin_date_idx是否太小以至于不能切片，若是的话，就
    # 加上步长。直到可以切片为止
    # 这一个是在original_date中的索引，之后还需要在每个个股时间索引上判断（因为可能有nan）
    while True:
        if original_date_idx - step2_pred_period - step2_slice_range < 0:
            original_date_idx += step2_slice_range
        else:
            break



    bar = progressbar.ProgressBar(maxval=stock_number, \
        widgets=[progressbar.Bar('>', '[', ']','-'), ' ',progressbar.Percentage(),' ' ,progressbar.ETA()],\
        term_width = 50\
        )
    bar.start()


    for stock_num_count in range(stock_number):
        bar.update(stock_num_count + 1)
        temp_x = factors[..., stock_num_count]
        temp_y = RET[..., stock_num_count]
        time_column = original_time_column
        # 判断某只股票在某日内的CLOSE（用6个基本数据中的任意一个都行）是否nan，如果是，将不用这一日的数据
        # 但是从机器学习4.0开始，由于我们去掉了上市、重新上市前以及CLOSE为负的一些数据，
        # 导致factors中的“原始数据”中存在的nan和6个基本原始数据的nan也不一样了，因此，我们
        # 将上面的CLOSE也一起处理了。
        not_nan_idx = np.where(np.isnan(CLOSE[...,stock_num_count])==False)[0]
        temp_x = temp_x[:,not_nan_idx].copy()
        temp_y = temp_y[:,not_nan_idx].copy()
        time_column = time_column[not_nan_idx]
        # 判断一只股票是否上市
        if len(time_column) == 0:
            print("\033[1;34m"+ 'There is no value to No.'+str(stock_num_count)+' stock in all year. Skip!' +"\033[0m")
            continue
        # 判断一只股票的否存在某个因子全为nan
        # temp_x[np.where(np.isnan(temp_x))] = np.nanmean(temp_x, axis=1)[np.where(np.isnan(temp_x))[0]]
        if any(np.isnan(np.nanmean(temp_x, axis=1))):
            print("\033[1;34mFor No."+str(stock_num_count)+' stock, in train year, at least one factor is all nan.'+\
                    ' And the nan factors index is:' +str(np.where(np.isnan(np.nanmean(temp_x, axis=1)))[0]) + '\033[0m')
            continue

        # 找到该股票训练年度的index
        train_year_idx = np.zeros([len(time_column)])
        for count_time_column in range(len(time_column)):
            if int(time_column[count_time_column][0:4]) in train_year:
                train_year_idx[count_time_column] = 1

        # 判断一只股票在train year是否有数据
        if len(train_year_idx) == 0:
            print("\033[1;34m"+ 'There is no value to No.'+str(stock_num_count)+' stock in train year. Skip!' +"\033[0m")
            continue

        # 有些因子有inf或nan，均填充为均值
        temp_x[np.where(np.isinf(temp_x))] = np.nan
        temp_x[np.where(np.isnan(temp_x))] = np.nanmean(temp_x, axis=1)[np.where(np.isnan(temp_x))[0]]

        # 如果数据量小于一次切片的数据量，跳过此次循环
        if len(np.where(train_year_idx==1)[0]) < step2_pred_period + step2_slice_range:
            continue

        #####
        # 上面部分都是对该支股票的数据进行处理。接下来就可以开始进行切片的操作。


        ### 开始切片 ###
        while True:
            # 结束循环条件
            if original_date_idx > train_end_date_idx:
                break
            else:
                # 利用original_date_idx找到对应在original_time_column中的所有
                # pred_period对应的索引。这样可以避免如果pred_period中出现nan时
                # 导致的错位问题
                original_date_in_pred_peirod = original_time_column[original_date_idx - step2_pred_period : original_date_idx]
                # 利用original_date_in_pred_peirod对应到time_column的日期，再找到其索引。
                i_in_pred_peirod = np.array([np.where( _ ==time_column)[0][0] for _ in original_date_in_pred_peirod if _ in time_column])

                # 如果pred_period中全为nan，就更新条件并继续下一次循环
                if len(i_in_pred_peirod) == 0:
                    original_date_idx += step2_slice_stride
                    continue

                # i代表预测周期中的最后一天
                i = i_in_pred_peirod[-1]
                # 判断此时的索引值是否足够切片，否则加上步长
                while True:
                    if i - len(i_in_pred_peirod) - step2_slice_range < 0:
                        i += step2_slice_stride
                    else:
                        break

                
                # 检查temp_y(即RET)在[i-len(i_in_pred_peirod), i]中有没有inf或nan
                check_y = np.cumprod(temp_y[:, i - len(i_in_pred_peirod) : i]+1)[-1] - 1
                if np.isinf(check_y) or np.isnan(check_y):
                        original_date_idx += step2_slice_stride     # 更新循环条件
                        continue

                # 标准化
                check_x = temp_x[:, i - len(i_in_pred_peirod) - step2_slice_range : i - len(i_in_pred_peirod)]
                mean = np.nanmean(check_x, axis=1).reshape(factor_number, 1)
                std = np.nanstd(check_x, axis=1, ddof=1).reshape(factor_number, 1)
                check_x = (check_x - mean)/std

                # 检查是否存在nan和inf
                if np.sum(np.isnan(check_x)) > 0:
                    print("\033[1;31m"+ "There are nan in x_train" + '\033[0m')
                    print("\033[1;31m"+ "It's the "+str(stock_num_count)+"th stock(index, begin with 0), and the last date is "+ original_date  + '\033[0m')
                    sys.exit()
                elif np.sum(np.isinf(check_x)) > 0:
                    print("\033[1;31m"+ "There are inf in x_train" + '\033[0m')
                    print("\033[1;31m"+ "It's the "+str(stock_num_count)+"th stock(index, begin with 0), and the last date is "+ original_date  + '\033[0m')
                    sys.exit()

                # 如果没有问题，就进行赋值
                x_train[count,...] = check_x.copy()
                y_train[count] = check_y
                # 更新计数
                count += 1
                # 更新循环条件
                original_date_idx += step2_slice_stride


    x_train = np.delete(x_train, range(count, x_train.shape[0]), axis=0)
    y_train = np.delete(y_train, range(count, y_train.shape[0]), axis=0)

    # 再次检查有没有nan
    if np.sum(np.isnan(x_train))>0:
        print("\033[1;31m"+'There are nan in x_train. CHECK!!!' + '\033[0m')
        sys.exit()

    # 保存训练数据
    np.savez(step2_save_dir+'/train.npz', \
                x_train = x_train.copy(),\
                y_train = y_train.copy(),\
                )

    print("\033[1;35m"+'train sample number: '+str(y_train.shape[0])+  '\033[0m')
    del x_train, y_train, check_x, check_y




    ####################################
    #############测试数据切片#############
    ####################################
    print('    Train year is '+str(test_year)+'...')
    original_test_year_idx = np.zeros([len(original_time_column)])
    for count_time_column in range(len(original_time_column)):
        if int(original_time_column[count_time_column][0:4]) in np.array(test_year):
            original_test_year_idx[count_time_column] = 1


    # 首先判断一下每个因子的nan比例
    total_size = factors.shape[1]*factors.shape[2]
    CLOSE_nan_percentage = np.sum(np.isnan(CLOSE))/total_size
    nan_percentage_of_each_factors = np.nan*np.zeros(factors.shape[0])
    for factor_num_count in range(factors.shape[0]):
        nan_percentage_of_each_factors[factor_num_count] = \
            np.sum(np.isnan(factors[factor_num_count,...]))/total_size

    # 对于nan比例大于0.05的因子，删掉
    # 如果没有nan大于临界值的因子，那么只需要删除未上市/停牌后的值，剩下的填充
    del_factors_idx = np.where((nan_percentage_of_each_factors-CLOSE_nan_percentage)>0.05)[0]
    if del_factors_idx.shape[0] != 0:
        print("\033[1;31mThere are factors NaN percentage greater than 0.05 reletive CLOSE" + '\033[0m')
        print("\033[1;31mTheir index are:" + str(del_factors_idx) + '\033[0m')
        print("\033[1;31mDeleting..." + '\033[0m')
        factors = np.delete(factors, del_factors_idx, axis=0)
    else:
        pass
       

    # //是向下取整除法
    # sample_num大于最大的样本数量，在有nan时可能填不满，因此有一些可能需要删除
    # 这里对nan的处理方法还是采用因子函数那样的方法。
    print('    Preparing testing samples...')
    sample_num = stock_number*((np.sum(original_test_year_idx==1)) // step2_slice_stride + 1)
    x_test = np.nan*np.zeros([sample_num, factor_number, step2_slice_range])
    y_test = np.nan*np.zeros(sample_num)
    time_recorder = []
    stock_recorder = []
    count = 0       # 用count对x_test和y_test计数


    # original_test_begin_date_idx是original_time_column的索引。original_time_column是没有
    # 经过nan修正的，因此可以作为所有股票的统一日程规范。经过每一次循环后，
    # original_test_begin_date_idx就加上step2_slice_stride, 并且在前step2_pred_period的索引范围内
    # 找某只股票有没有对应的日期。有的话就定下这几天内的累计收益率，并且
    # 切片original_test_begin_date_idx - step2_pred_period 前step2_slice_range天（经过nan）修正后
    # 的数据为x

    # 所有的xxx_date_idx都将是样本的累计收益率的最后一天的索引
    # 首先通过original_time_column定位每一次切片的日期。这样可以保证在step2_slice_stride ！= 1
    # 时，切片不会由于nan的出现而错位。这样对之后evaluation部分再换仓部分的计算比较便利
    original_test_begin_date_idx_set = np.zeros(original_time_column.shape)
    for original_test_begin_date_idx in range(len(original_time_column)):
        if int(original_time_column[original_test_begin_date_idx][0:4]) in np.array(test_year):
            original_test_begin_date_idx_set[original_test_begin_date_idx] = 1


    test_begin_date_idx = np.min(np.where(original_test_begin_date_idx_set==1)[0])
    test_end_date_idx = np.max(np.where(original_test_begin_date_idx_set==1)[0])+1



    # 确定original中的索引起始点:
    original_date_idx = test_begin_date_idx + step2_pred_period  

    # 判断当前的original_test_begin_date_idx是否太小以至于不能切片，若是的话，就
    # 加上步长。直到可以切片为止
    # 这一个是在original_date中的索引，之后还需要在每个个股时间索引上判断（因为可能有nan）
    while True:
        if original_date_idx - step2_pred_period - step2_slice_range < 0:
            original_date_idx += step2_slice_range
        else:
            break


    bar = progressbar.ProgressBar(maxval=stock_number, \
        widgets=[progressbar.Bar('>', '[', ']','-'), ' ',progressbar.Percentage(),' ' ,progressbar.ETA()],\
        term_width = 50\
        )
    bar.start()


    for stock_num_count in range(stock_number):
        bar.update(stock_num_count + 1)
        temp_x = factors[..., stock_num_count]
        temp_y = RET[..., stock_num_count]
        time_column = original_time_column
        # 判断某只股票在某日内的CLOSE（用6个基本数据中的任意一个都行）是否nan，如果是，将不用这一日的数据
        # 但是从机器学习4.0开始，由于我们去掉了上市、重新上市前以及CLOSE为负的一些数据，
        # 导致factors中的“原始数据”中存在的nan和6个基本原始数据的nan也不一样了，因此，我们
        # 将上面的CLOSE也一起处理了。
        not_nan_idx = np.where(np.isnan(CLOSE[...,stock_num_count])==False)[0]
        temp_x = temp_x[:,not_nan_idx].copy()
        temp_y = temp_y[:,not_nan_idx].copy()
        time_column = time_column[not_nan_idx]
        # 判断一只股票是否上市
        if len(time_column) == 0:
            print("\033[1;34m"+ 'There is no value to No.'+str(stock_num_count)+' stock in all year. Skip!' +"\033[0m")
            continue
        # 判断一只股票的否存在某个因子全为nan
        # temp_x[np.where(np.isnan(temp_x))] = np.nanmean(temp_x, axis=1)[np.where(np.isnan(temp_x))[0]]
        if any(np.isnan(np.nanmean(temp_x, axis=1))):
            print("\033[1;34mFor No."+str(stock_num_count)+' stock, in test year, at least one factor is all nan.'+\
                    ' And the nan factors index is:' +str(np.where(np.isnan(np.nanmean(temp_x, axis=1)))[0]) + '\033[0m')
            continue

        # 找到该股票测试年度的index
        test_year_idx = np.zeros([len(time_column)])
        for count_time_column in range(len(time_column)):
            if int(time_column[count_time_column][0:4]) in np.array(test_year):
                test_year_idx[count_time_column] = 1

        # 判断一只股票在test year是否有数据
        if len(test_year_idx) == 0:
            print("\033[1;34m"+ 'There is no value to No.'+str(stock_num_count)+' stock in test year. Skip!' +"\033[0m")
            continue

        # 有些因子有inf或nan，均填充为均值
        temp_x[np.where(np.isinf(temp_x))] = np.nan
        temp_x[np.where(np.isnan(temp_x))] = np.nanmean(temp_x, axis=1)[np.where(np.isnan(temp_x))[0]]

        # 如果数据量小于一次切片的数据量，跳过此次循环
        if len(np.where(test_year_idx==1)[0]) < step2_pred_period + step2_slice_range:
            continue

        #####
        # 上面部分都是对该支股票的数据进行处理。接下来就可以开始进行切片的操作。


        ### 开始切片 ###
        while True: 
            # 结束循环条件
            if original_date_idx > test_end_date_idx:
                break
            else:
                # 利用original_date_idx找到对应在original_time_column中的所有
                # pred_period对应的索引。这样可以避免如果pred_period中出现nan时
                # 导致的错位问题
                original_date_in_pred_peirod = original_time_column[original_date_idx - step2_pred_period : original_date_idx]
                # 利用original_date_in_pred_peirod对应到time_column的日期，再找到其索引。
                i_in_pred_peirod = np.array([np.where( _ ==time_column)[0][0] for _ in original_date_in_pred_peirod if _ in time_column])

                # 如果pred_period中全为nan，就更新条件并继续下一次循环
                if len(i_in_pred_peirod) == 0:
                    original_date_idx += step2_slice_stride
                    continue

                # i代表预测周期中的最后一天
                i = i_in_pred_peirod[-1]
                # 判断此时的索引值是否足够切片，否则加上步长
                while True:
                    if i - len(i_in_pred_peirod) - step2_slice_range < 0:
                        i += step2_slice_stride
                    else:
                        break

                
                # 检查temp_y(即RET)在[i-len(i_in_pred_peirod), i]中有没有inf或nan
                check_y = np.cumprod(temp_y[:, i - len(i_in_pred_peirod) : i]+1)[-1] - 1
                if np.isinf(check_y) or np.isnan(check_y):
                        original_date_idx += step2_slice_stride     # 更新循环条件
                        continue

                # 标准化
                check_x = temp_x[:, i - len(i_in_pred_peirod) - step2_slice_range : i - len(i_in_pred_peirod)]
                mean = np.nanmean(check_x, axis=1).reshape(factor_number, 1)
                std = np.nanstd(check_x, axis=1, ddof=1).reshape(factor_number, 1)
                check_x = (check_x - mean)/std

                # 检查是否存在nan和inf
                if np.sum(np.isnan(check_x)) > 0:
                    print("\033[1;31m"+ "There are nan in x_test" + '\033[0m')
                    print("\033[1;31m"+ "It's the "+str(stock_num_count)+"th stock(index, begin with 0), and the last date is "+ original_date  + '\033[0m')
                    sys.exit()
                elif np.sum(np.isinf(check_x)) > 0:
                    print("\033[1;31m"+ "There are inf in x_test" + '\033[0m')
                    print("\033[1;31m"+ "It's the "+str(stock_num_count)+"th stock(index, begin with 0), and the last date is "+ original_date  + '\033[0m')
                    sys.exit()

                # 如果没有问题，就进行赋值
                x_test[count,...] = check_x.copy()
                y_test[count] = check_y
                stock_recorder.append(stock_num_count)
                time_recorder.append(original_date_in_pred_peirod[-1])
                # 更新计数 
                count += 1
                # 更新循环条件
                original_date_idx += step2_slice_stride


    x_test = np.delete(x_test, range(count, x_test.shape[0]), axis=0)
    y_test = np.delete(y_test, range(count, y_test.shape[0]), axis=0)

    # 再次检查有没有nan
    if np.sum(np.isnan(x_test))>0:
        print("\033[1;31m"+'There are nan in x_test. CHECK!!!' + '\033[0m')
        sys.exit()


    # 保存测试数据
    np.savez(step2_save_dir+'/test.npz', \
                x_test = x_test.copy(),\
                y_test = y_test.copy(),\
                time_recorder = time_recorder.copy(),\
                stock_recorder = stock_recorder.copy())


    print("\033[1;35m"+'test sample number: '+str(y_test.shape[0]) + '\033[0m')

    print('Ending at: ',time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())))

