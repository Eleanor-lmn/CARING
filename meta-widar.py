"""
    meta-test.py 使用元学习方法训练单独用户无线手势识别
    2021.7.15
"""
# -*- coding: utf-8 -*-
import random
from typing import Any
from keras import utils
from keras.models import Input, Model
from keras.layers import BatchNormalization, Activation, MaxPooling2D, Lambda
from keras import backend as K
import os
import numpy as np
from keras import optimizers
from keras import layers, models
import math
from keras.optimizers import Adam
import copy

######################################################################################
# Meta- learning:
# for rouond = 1,2,3,.. do
#       Randdoly select a subset U for users.
#       for User j in U do
#           Pull model parameters Wc from server.
#           Train m epochs of the embedding network on local dataset Dj through
#            pairwise loss, get locally updated parameters Wj.
#           Push the updated parameters Wj to server.
#       FL server updated central model: Wc <- Wc + k(Wmean - Wc)
# 服务器参数更新：对测试数据有利的参数更新才可以传递给服务器
# 测试用户分为3部分：label、gradient test、unlabel = 1：1：3
######################################################################################

def init_server_System():
    # 各组件使用deepSense作为框架
    ### 相关参数
    image_input_shape = (90, 999, 1)  # 输入图片维度
    num_classes = 9  # 数据类别
    # ------------------------- 架构 = DeepSense↓ ------------------------------
    input_tensor = Input(shape=image_input_shape)
    x = layers.Reshape((90, 9, 111))(input_tensor)
    client_conv1 = layers.Conv2D(128, kernel_size=(7, 7), activation='relu', padding='same', name='Feature_conv_1')(
        x)
    client_bn1 = layers.BatchNormalization(momentum=0.8, name='Feature_batchNorm_1')(client_conv1)
    client_drop1 = layers.Dropout(0.3, name='Feature_drop_1')(client_bn1)

    client_conv2 = layers.Conv2D(64, kernel_size=(7, 7), activation='relu', padding='same', name='Feature_conv_2')(
        client_drop1)
    client_bn2 = layers.BatchNormalization(momentum=0.8, name='Feature_batchNorm_2')(client_conv2)
    client_drop2 = layers.Dropout(0.3, name='Feature_drop_2')(client_bn2)
    # ------------------------ LSTM 分类 ↓ ------------------------------------
    flatten_1 = layers.Flatten(name="Classify_Flatten")(client_drop2)
    reshape_1 = layers.Reshape((90, 576), name='Classify_Reshape')(flatten_1)
    bilstm_1 = layers.Bidirectional(layers.LSTM(32, unroll=True, return_sequences=True), name= 'BiLSTM_1')(reshape_1)
    bilstm_2 = layers.Bidirectional(layers.LSTM(16, unroll=True, return_sequences=True), name="BiLSTM_2")(bilstm_1)
    flatten_2 = layers.Flatten(name="Classify_Flatten_2")(bilstm_2)
    dense_1 = layers.Dense(128, activation='relu', name="Classify_Dense")(flatten_2)

    classify_output = layers.Dense(num_classes, activation='softmax', name="Classify_Output")(dense_1)

    return Model(input_tensor, classify_output)


def build_meta_System():
    # 各组件使用deepSense作为框架
    ### 相关参数
    image_input_shape = (90, 999, 1) # 输入图片维度
    num_classes = 9 # 数据类别
    # ------------------------- 架构 = DeepSense↓ ------------------------------
    input_tensor = Input(shape=image_input_shape)
    x = layers.Reshape((90, 9, 111))(input_tensor)
    x = layers.Conv2D(128, kernel_size=(7, 7), activation='relu', padding='same', name='Feature_conv_1')(x)
    client_bn1 = layers.BatchNormalization(momentum=0.8, name='Feature_batchNorm_1')(x)
    client_drop1 = layers.Dropout(0.3, name='Feature_drop_1')(client_bn1)

    client_conv2 = layers.Conv2D(64, kernel_size=(7, 7), activation='relu', padding='same', name='Feature_conv_2')(
        client_drop1)
    client_bn2 = layers.BatchNormalization(momentum=0.8, name='Feature_batchNorm_2')(client_conv2)
    client_drop2 = layers.Dropout(0.3, name='Feature_drop_2')(client_bn2)
    # ------------------------ LSTM 分类 ↓ ------------------------------------
    flatten_1 = layers.Flatten(name="Classify_Flatten_1")(client_drop2)
    reshape_1 = layers.Reshape((90,576),name='Classify_Reshape')(flatten_1)
    bilstm_1 = layers.Bidirectional(layers.LSTM(32, unroll=True, return_sequences=True), name= 'BiLSTM_1')(reshape_1)
    bilstm_2 = layers.Bidirectional(layers.LSTM(16, unroll=True, return_sequences=True), name="BiLSTM_2")(bilstm_1)
    flatten_2 = layers.Flatten(name="Classify_Flatten_2")(bilstm_2)
    dense_1 = layers.Dense(128, activation='relu', name="Classify_Dense")(flatten_2)

    classify_output = layers.Dense(num_classes, activation='softmax', name="Classify_Output")(dense_1)

    return Model(input_tensor, classify_output)


#  根据（路径，样本总数，类别个数）参数读取所有txt文件内容，默认各类数据个数相同，不需修改
def Load_Txt_Data(path, Sum_Samples, CategoryNumber):
    # 一个种类的数据个数
    OneCategoryNumber = Sum_Samples // CategoryNumber
    # 读取训练数据路径文件名
    paths = []
    for i in range(CategoryNumber):
        for j in range(OneCategoryNumber):
            file = (path + '/%s.txt' % (i * OneCategoryNumber + j + 1))
            paths.append(file)
    imgs = []
    # 每次i指向下一个图片
    for i in range(len(paths)):
        img = Get_Path_TxtData(paths[i])
        imgs.append(img)
    # 转化为array
    imgs = np.array(imgs, dtype=float)
    return imgs

#  根据路径参数读取一个txt文件内容（两列数值型txt文件，eg：RFID数据），不需修改
def Get_Path_TxtData(path):
    image = []
    with open(path, 'r') as f:
        # 读取所有数据
        lines = f.readlines()
        for line in lines:
            words = line.split(", ")
            words = words[:1]
            image.append(words)
        images = np.array(image, dtype=float)
        ## 按最大最小值归一化
        imax = np.max(images)
        imin = np.min(images)
        images = 2 * (images - imin) / (imax - imin) - 1
    return images

# 根据（类别数，每类数据个数（每个种类数据个数相同））参数生成txt文件数据的标签，不需修改
def Get_Labels(CategoryNumber, OneCategoryNum):
    labels = []
    for i in range(CategoryNumber):
        for j in range(OneCategoryNum):
            labels.append(i)
    labels = np.array(labels, dtype=float)
    return labels

def load_user_data(CategoryNumber, path, sample):  # 数据类别数, 路径，
    # 训练数据
    data = np.zeros(shape=(1, 999, 1))
    label = np.zeros(shape=(1,))
    trainX1 = Load_Txt_Data(path, sample * CategoryNumber, CategoryNumber)  # 路径 数据总数 类别数
    trainY1 = Get_Labels(CategoryNumber, sample // 90)
    data = np.concatenate((data, trainX1), axis=0)
    label = np.concatenate((label, trainY1), axis=0)
    data = data[1:, :, :]
    data = data.reshape(data.shape[0] // 90, 90, 999, 1)
    # data = np.expand_dims(data, axis=3)
    label = label[1:]
    return data, label

def update_server_weights(w_list, w_pram, w, sigma=0.2):
    """
    model 1 and model 2 with same structure
    return weights dict with values w_model1 - w_model2

    sigma = 1 : federated learning
    """
    w_avg = copy.deepcopy(np.multiply(w_list[0], w_pram[0]))
    num = 0
    for i in range(len(w_pram)):
        num += w_pram[i]
    for k in range(len(w)):
        for idx in range(1, len(w_list)):
            w_avg[k] += np.multiply(w_list[idx][k], w_pram[idx])
        w_avg[k] = w[k] + np.multiply(np.subtract(np.divide(w_avg[k], num), w[k]), sigma)

    return w_avg

def update_user_weights(w_user, w, sigma=0.2):
    """
    model 1 and model 2 with same structure
    return weights dict with values w_model1 - w_model2

    sigma = 1 : federated learning
    """
    for k in range(len(w)):
        w_user[k] = w[k] + np.multiply(np.subtract(w_user[k], w[k]), sigma)

    return w_user

def train():
    #####################训练参数
    rounds = 1000
    init_server_rounds = 3  ## 初始化server训练轮数
    client_rounds = 3

    server_user = 3  ## 服务器数据
    client_train = 7  ## 已知用户
    client_test = 2  ## 新用户
    CategoryNumber = 9 ## 类别数
    batch_sizes = 128
    sigma = 0.1
    # 160条数据：10 * 16
    actionNum = 5
    fileName = './meta-test3_widar999_c.txt'
    modelName = './widar999/'
    update_weight = 0.3
    round_times = 90
    #os.makedirs(modelName)

    # -----------------------------------------------------------------------
    # -------------------------- 数据准备 -----------------------------------
    # -----------------------------------------------------------------------
    #################### WiFi数据准备2 (读取前n个为有label，后20-n个为无label，且将wulabel按照不同次实验分为训练集和测试集)
    WiFiDataPath = "/home/SL/CrossGR2/classifyModel/WidarDatasets/Widar3_new"
    sum_sample = 450  # 90 子载波

    ################### client_train_models ######################
    client_models = []  # type: Any
    for ind in range(client_train):
        client_model = build_meta_System()
        # 优化器
        C_adam = Adam(lr=0.0001, decay=1e-6)
        # loss配置: loss使用列表的形式,要与模型建立的多输出分别对应; loss_weights指定两个loss优化时的权重
        client_model.summary()
        client_model.compile(loss='sparse_categorical_crossentropy', optimizer=C_adam, # 交叉熵损失函数
                          metrics=['accuracy'])  # ,loss_weights=[1, 0.1]
        client_models.append(client_model)

    ################## client_test_models #######################
    client_test_models = []
    for ind in range(client_test):
        client_model = build_meta_System()
        # 优化器
        C_adam = Adam(lr=0.0001, decay=1e-6)
        # loss配置: loss使用列表的形式,要与模型建立的多输出分别对应; loss_weights指定两个loss优化时的权重
        client_model.summary()
        client_model.compile(loss='sparse_categorical_crossentropy', optimizer=C_adam, # 交叉熵损失函数
                          metrics=['accuracy'])  # ,loss_weights=[1, 0.1]
        client_test_models.append(client_model)

    ################## 初始化 server ###########################
    #### -----------------------读入服务器数据-------------------------------- ####
    server_data = np.zeros(shape=(1, 90, 999, 1))
    server_label = np.zeros(shape=(1,))
    for i in range(server_user):
        path = WiFiDataPath + "/InterMean" + str(i)
        serverData, serverLabel = load_user_data(CategoryNumber, path, sum_sample)
        server_data = np.concatenate((server_data, serverData), axis=0)
        server_label = np.concatenate((server_label, serverLabel), axis=0)
    server_data = server_data[1:, :, :, :]
    # server_data = np.expand_dims(server_data, axis=3)
    server_label = server_label[1:]
    print("server_data.shape: ", server_data.shape)
    print("server_label.shape: ", server_label.shape)
    #### ----------------------初始化服务器----------------------------------####
    server_model = init_server_System()
    # 优化器
    C_adam = Adam(lr=0.0001, decay=1e-6)
    # loss配置: loss使用列表的形式,要与模型建立的多输出分别对应; loss_weights指定两个loss优化时的权重
    server_model.summary()
    server_model.compile(loss='sparse_categorical_crossentropy', optimizer=C_adam,  # 交叉熵损失函数
                         metrics=['accuracy'])  # ,loss_weights=[1, 0.1]
    for i in range(init_server_rounds):
        idx_train_class = random.sample(range(0, server_data.shape[0]), batch_sizes)
        batch_server_data = server_data[idx_train_class]
        batch_server_label = server_label[idx_train_class]
        server_param = server_model.train_on_batch(batch_server_data, batch_server_label)
        print("Training Server --- loss: %.2f ---  classify_acc: %.2f%%" % (server_param[0], server_param[1] * 100))
        with open(fileName, 'a') as f:
            f.write("Training Server --- loss: %.2f ---  classify_acc: %.2f%%\n" % (server_param[0], server_param[1] * 100))

    # server_model.save_weights('server_model.h5')
    ###################### 训练用户 client ##########################################
    
    ##################### 服务器参数更新 ##########################################
    # 服务器参数更新：对测试数据有利的参数更新才可以传递给服务器
    # 设置一个初始的测试精度为0
    accuracy = 0

    # 测试用户分为3类：label、gradient test、unlabel = 1：1：3
    idx_train_class_1 = [i * actionNum + 0 for i in range(CategoryNumber)]     # label
   

    idx_train_class_2 = [i * actionNum + 2 for i in range(CategoryNumber)]     # gradient test
    

    idx_train_class_3 = idx_train_class_1 + idx_train_class_2                  # unlabel


    for round in range(rounds):
        # 随机选择训练用户：服务器数据为1-3，训练用户在4-8中选3个，测试用户2
        chosen_client = np.random.choice(client_train, client_train, replace=False)
        chosen_client_test = np.random.choice(client_test, client_test, replace=False) + client_train

        #### --------------------------- 导入数据 --------------------------------####
        client_users_data = []
        client_users_label = []
        for clientInd in (chosen_client):  ## 训练数据导入
            path = WiFiDataPath + "/InterMean" + str(clientInd)
            clientData, clientLabel = load_user_data(CategoryNumber, path, sum_sample)
            client_users_data.append(clientData)
            client_users_label.append(clientLabel)

        client_test_users_data = []
        client_test_users_label = []
        for clientTestInd in (chosen_client_test):  ## 测试数据导入
            path = WiFiDataPath + "/InterMean" + str(clientTestInd)
            clientTestData, clientTestLabel = load_user_data(CategoryNumber, path, sum_sample)
            client_test_users_data.append(clientTestData)
            client_test_users_label.append(clientTestLabel)
        #### ------------------- 本地训练 每个client分别更新 -----------------------####
        update_weights = []
        update_weights_pram = []
        for user in range(len(chosen_client)):
            # pull weights theta from center
            ###############################
            print("      -- Start local training on user:  user", user)
            with open(fileName, 'a') as f:
                f.write("      -- Start local training on user:  user" + str(user) + "\n")
            # print(server_model.get_weights())
            # client_models[user].load_weights('server_model.h5')
            client_models[user].set_weights(server_model.get_weights())
            # local train
            for j in range(client_rounds):
                # idx_train_class = random.sample(range(0, client_users_data[user].shape[0]), batch_sizes)
                # batch_client_data = client_users_data[user][idx_train_class]
                # batch_client_label = client_users_label[user][idx_train_class]
                batch_client_data = client_users_data[user][:]
                batch_client_label = client_users_label[user][:]
                client_param = client_models[user].train_on_batch(batch_client_data, batch_client_label)
                print( "[%d/%d] Training client %d --- loss: %.2f --- classify_acc: %.2f%%" % (round, rounds, user, client_param[0], client_param[1] * 100))
                with open(fileName, 'a') as f:
                    f.write( "[%d/%d] Training client %d --- loss: %.2f --- classify_acc: %.2f%%\n" % (round, rounds, user, client_param[0], client_param[1] * 100))
            
            ## 判断当前用户的参数更新是否传递向服务器，参数更新对新用户有利时，才将参数更新传到服务器
            ## 选择一部分测试数据进行梯度判断
            accu = 0
            for user_idx in range(len(chosen_client_test)):
                batch_client_data = client_test_users_data[user_idx][idx_train_class_2]
                batch_client_label = client_test_users_label[user_idx][idx_train_class_2]

                # 当前用户更新后的服务器参数权值
                weight_current = update_user_weights(client_models[user].get_weights(), server_model.get_weights(), sigma=sigma)
                client_test_models[user_idx].set_weights(weight_current)
                client_param = client_test_models[user_idx].train_on_batch(batch_client_data, batch_client_label)

                print("[%d/%d] Testing Labelled Data user%d --- loss: %.2f --- classify_acc: %.2f%%" % (round, rounds, user_idx, client_param[0], client_param[1] * 100))
                with open(fileName, 'a') as f:
                    f.write("[%d/%d] Testing Labelled Data user%d --- loss: %.2f --- classify_acc: %.2f%%\n" % (round, rounds, user_idx, client_param[0], client_param[1] * 100))
                accu += client_param[1]
            
            # 参数更新传递向服务器
            update_weights.append(client_models[user].get_weights())
            #####  更新参数的权重
            accuracy_current = accu / client_test
            if round > round_times and accuracy_current < accuracy:
                update_weights_pram.append(update_weight)
                print("update weight : %.1f!" %update_weight)
                with open(fileName, 'a') as f:
                    f.write("update weight : %.1f!\n" %update_weight)
            else:
                update_weights_pram.append(1.0)
                print("update weight : 1.0!")
                with open(fileName, 'a') as f:
                    f.write("update weight : 1.0!\n")

            ## user train Done
            print("Done local train on user: user%s" % user)
            with open(fileName, 'a') as f:
                f.write("Done local train on user: user%s\n" % user)

        #### ----------------------- update global meta model --------------------------- ####
        print("----------------------- update global meta model ---------------------------")
        server_model.set_weights(update_server_weights(update_weights, update_weights_pram, server_model.get_weights(), sigma=sigma))
        server_model.save_weights(modelName + str(round) + '.h5')
        print("Done update global meta model")
        with open(fileName, 'a') as f:
            f.write("Done update global meta model\n")


        test_update_weights = []
        test_update_weights_pram = []
        print("# ====== Testing Client Train ===========")
        with open(fileName, 'a') as f:
            f.write("# ====== Testing Client Train ===========\n")
        for user_idx in range(len(chosen_client)):
            client_models[user_idx].set_weights(server_model.get_weights())
            acc = client_models[user_idx].evaluate(client_users_data[user_idx], client_users_label[user_idx])
            print("[%d/%d] Client Train user%d --- loss: %.2f --- ACC: %.2f%%" % (round, rounds, user_idx, acc[0], acc[1] * 100))
            with open(fileName, 'a') as f:
                f.write("[%d/%d] Client Train user%d --- loss: %.2f --- ACC: %.2f%%\n" % (round, rounds, user_idx, acc[0], acc[1] * 100))


        print("# ====== Testing Client_Test ===========")
        with open(fileName, 'a') as f:
            f.write("# ====== Testing Client_Test ===========\n")

        accu = 0
        for user_idx in range(len(chosen_client_test)):
            # 测试用户的label数据
            batch_client_data = client_test_users_data[user_idx][idx_train_class_1]
            batch_client_label = client_test_users_label[user_idx][idx_train_class_1]
            client_test_models[user_idx].set_weights(server_model.get_weights())
            client_param = client_test_models[user_idx].train_on_batch(batch_client_data, batch_client_label)
            #####  更新权值
            test_update_weights.append(client_test_models[user_idx].get_weights())
            test_update_weights_pram.append(1.0)

            print("[%d/%d] Testing Labelled Data user%d --- loss: %.2f --- classify_acc: %.2f%%" % (round, rounds, user_idx, client_param[0], client_param[1] * 100))
            with open(fileName, 'a') as f:
                f.write("[%d/%d] Testing Labelled Data user%d --- loss: %.2f --- classify_acc: %.2f%%\n" % (round, rounds, user_idx, client_param[0], client_param[1] * 100))
            
            # 测试用户的test gradient数据
            batch_client_data = client_test_users_data[user_idx][idx_train_class_2]
            batch_client_label = client_test_users_label[user_idx][idx_train_class_2]
            client_test_models[user_idx].set_weights(server_model.get_weights())
            client_param = client_test_models[user_idx].train_on_batch(batch_client_data, batch_client_label)
            accu += client_param[1]

            print("[%d/%d] Testing gradient Data user%d --- loss: %.2f --- classify_acc: %.2f%%" % (round, rounds, user_idx, client_param[0], client_param[1] * 100))
            with open(fileName, 'a') as f:
                f.write("[%d/%d] Testing gradient Data user%d --- loss: %.2f --- classify_acc: %.2f%%\n" % (round, rounds, user_idx, client_param[0], client_param[1] * 100))

            # 测试用户的unlabel数据
            batch_client_data_test = np.delete(client_test_users_data[user_idx], idx_train_class_3, axis=0)
            batch_client_label_test = np.delete(client_test_users_label[user_idx], idx_train_class_3, axis=0)
            acc = client_test_models[user_idx].evaluate(batch_client_data_test, batch_client_label_test)

            print("[%d/%d] Testing Unlabelled Data user%d --- loss: %.2f --- ACC: %.2f%%" % (round, rounds, user_idx, acc[0], acc[1] * 100))
            with open(fileName, 'a') as f:
                f.write("[%d/%d] Testing Unlabelled Data user%d --- loss: %.2f --- ACC: %.2f%%\n" % (round, rounds, user_idx, acc[0], acc[1] * 100))
            
        accuracy = accu / client_test
        # update global meta model(test)
        print("----------------------- update global meta model ---------------------------")
        server_model.set_weights(update_server_weights(test_update_weights, test_update_weights_pram, server_model.get_weights(), sigma=sigma))


if __name__ == '__main__':
    train()
