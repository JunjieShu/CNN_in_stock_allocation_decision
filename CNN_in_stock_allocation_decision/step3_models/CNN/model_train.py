import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import regularizers
from keras import backend as K
import os
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler # 归一化
import tensorflow.gfile as gfile
from keras.models import model_from_json     # 只将模型结构保存为json文件
import sys
# sys.path.append("../../")     #用于增加函数搜索路径
# from s0_parameters import *

def model_train(test_year, train_period, Continue_model=False, epoch=3, step2_Dataset = 3, 
                  step2_slice_range = 60, step2_slice_stride = 1, chosen_stocks = 500,
                  step2_pred_period = 1,  evaluate_test = True,
                  ):

    # test_year=2016
    # train_period=1
    # epoch = 20
    # step2_Dataset = 1
    # step2_slice_range = 60
    # step2_slice_stride = 5
    # step2_pred_period = 1
    # chosen_stocks = 500

    # ==============

    print('Beginning at: ',time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())))


    ### 设定存、读路径
    step3_read_dir = '../../../dataset/training_data/'+'Dataset'+\
                      str(step2_Dataset)+'-stock_number'+str(chosen_stocks)+'-TestYear'+str(test_year)+\
                      '-TrainPeriod'+str(train_period) +'-'+'SliceRange'+\
                      str(step2_slice_range) + '-stride'+str(step2_slice_stride) +\
                       '-pred'+str(step2_pred_period)
    step3_model_save_dir = "./history"
    step3_model_name = step3_model_save_dir+'/Dataset'+str(step2_Dataset)+'-stock_number'+str(chosen_stocks)+\
                        '-TestYear'+str(test_year)+'-TrainPeriod'+\
                        str(train_period) +'-'+'SliceRange'+str(step2_slice_range) +\
                         '-stride'+str(step2_slice_stride) + '-pred'+str(step2_pred_period)

    ###文件读取
    print('reading_data')
    f = np.load(step3_read_dir + '/train.npz')
    x_train, y_train = f['x_train'], f['y_train']
    f.close()

    f = np.load(step3_read_dir + '/test.npz')
    x_test, y_test = f['x_test'], f['y_test']
    f.close()


    ### 定义输入尺寸；并将训练、测试样本整理到符合模型使用的样子

    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    ## 对y的处理步骤：
    # 1.对极端值进行处理；2.归一化

    y_train[np.where(y_train>0.1)] = 0.1
    y_train[np.where(y_train<-0.1)] = -0.1
    y_train = (y_train-np.mean(y_train))/np.std(y_train, ddof=1)
    y_train = (y_train - np.min(y_train))/(np.max(y_train) - np.min(y_train))


    y_test[np.where(y_test>0.1)] = 0.1
    y_test[np.where(y_test<-0.1)] = -0.1
    y_test = (y_test-np.mean(y_test))/np.std(y_test, ddof=1)
    y_test = (y_test - np.min(y_test))/(np.max(y_test) - np.min(y_test))


    ### 2. 通过Keras的API定义卷机神经网络
    # # 继续训练模型
    if Continue_model:
        print('Loading model...')
    # model = load_model(step4_model_path)

    # 使用Keras API定义模型：
    print('Creating model...')
    model = Sequential()

    model.add(Conv2D(16, kernel_initializer='glorot_normal', kernel_size=(3, 3), \
                      activation='relu', input_shape=input_shape))
    # model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(32, kernel_initializer='glorot_normal', kernel_size=(2, 2), \
                      activation='relu'))
    model.add(Conv2D(64, kernel_initializer='glorot_normal', kernel_size=(2, 2), \
                      activation='relu'))
    model.add(Conv2D(128, kernel_initializer='glorot_normal', kernel_size=(2, 2), \
                      activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Conv2D(64, kernel_initializer='glorot_normal', kernel_size=(2, 2), \
    #                   activation='relu'))

    model.add(Flatten())
    model.add(Dense(750, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dense(500, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='glorot_normal', activation='sigmoid'))



    # ### 查看模型网络结构
    # for lay in model.layers:
    #     print(lay.name)
    #     print(lay.get_weights())
    #     print('\n')

    ### 模型编译
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',
                # optimizer=keras.optimizers.RMSprop(lr=0.001),
                optimizer=sgd,
                metrics=['mae']      # 这个显示的是命令行中显示的信息
                )

    ### 3. 给出训练数据、batch大小、训练轮数和验证数据，Keras可以自动完成模型的训练过程
    # 训练模型，并将指标保存到 history 中
    # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    print('Training model...')
    history = model.fit(x_train, 
                        y_train,
                        batch_size=128,
                        epochs=epoch,
                        # 用于在控制台输出训练信息 0 = silent, 1 = progress bar, 
                        # 2 = one line per epoch:
                        verbose = 1,     

                        # validation数据比例:
                        validation_data = (x_test, y_test)
                      )



    ###  保存模型
    print('Saving model...')
    if gfile.Exists(step3_model_save_dir)==False:
        gfile.MakeDirs(step3_model_save_dir)

    model.save(step3_model_name)
    print('Saved trained model at %s ' % step3_model_name)


    # 模型在测试样本上的整体表现
    if evaluate_test:
        print('Evaluating model under test samples')
        loss_and_metrics = model.evaluate(x=x_test, y=y_test, 
                          # batch_size=None, 
                          verbose=1, 
                          # sample_weight=None, 
                          # steps=None, 
                          # allbacks=None
                          )
        print("\033[1;34mTest ame {}".format(loss_and_metrics[1]) + '\033[0m')
      
    print('Ending at: ',time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())))

    if evaluate_test:
        return loss_and_metrics[1]