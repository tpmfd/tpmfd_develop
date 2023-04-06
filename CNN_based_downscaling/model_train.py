import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.files import *
import myconv
import numpy as np
import datetime
import calendar
import ctypes

import os
import random
import sys
import time
from sys import exit as _exit
from sys import platform as _platform

import gdal,osr
import xlrd


def read_tif(in_path):

    dataset = gdal.Open(in_path)
    if dataset == None:
        print(in_path + " cannot be opened")
        return
    width = dataset.RasterXSize  # 栅格矩阵的列数
    height = dataset.RasterYSize  # 栅格矩阵的行数
    bands = dataset.RasterCount  # 波段数
    data = dataset.ReadAsArray(0, 0, width, height)  # 获取数据
    return data


def read_pre(rr):
    if rr == 2:
        res1 = 'coarse'
        res2 = 'median'
        pre1 = np.empty((59, 168, 730))  ###Change here
        pre2 = np.empty((118, 336, 730))
    elif rr == 4:
        res1 = 'median'
        res2 = 'fine'
        pre1 = np.empty((118, 336, 730))  ###Change here
        pre2 = np.empty((472, 1344, 730))
    else:
        print('The ratio of downscaling should be defined')
        sys.exit()
    k = 0
    for i in (2013, 2018):
        n_step = 365
        t_step = 0
        date1 = datetime.datetime(i, 1, 1, 0, 0, 0, 0)
        while t_step < n_step:
            date2 = date1 + datetime.timedelta(days=t_step)
            t_str = str(date2.strftime('%Y%m%d'))
            path1 = '/input_dir/'  + res1 + '/var_' + t_str + '.tif'
            print(path1)
            pre0 = read_tif(path1)
            pre1[:, :, k] = pre0
            k = k + 1
            t_step = t_step + 1
    k = 0
    for i in (2013, 2018):
        n_step = 365
        t_step = 0
        date1 = datetime.datetime(i, 1, 1, 0, 0, 0, 0)
        while t_step < n_step:
            date2 = date1 + datetime.timedelta(days=t_step)
            t_str = str(date2.strftime('%Y%m%d'))
            path1 = '/input_dir/' + res2 + '/var_' + t_str + '.tif'
            print(path1)
            pre0 = read_tif(path1)
            pre2[:, :, k] = pre0
            k = k + 1
            t_step = t_step + 1

    return pre1, pre2

def data_prepare(iy_c1,iy_c2,ix_c1,ix_c2,
                 iy_m1,iy_m2,ix_m1,ix_m2,
                 iy_f1,iy_f2,ix_f1,ix_f2,
                 ele_c,ele_std_c,
                 std_c,std_std_c,
                 ele_m,ele_std_m,
                 std_m,std_std_m,
                 ele_f,ele_std_f,
                 std_f,std_std_f,
                 rain_c,rain_std_c,
                 tile,rr,pre1,pre2):
    ################################
    ####### data processing

    # load topography data
    ele_coarse = np.loadtxt('/static_data/ele26664.asc')
    ele_coarse = ele_coarse[iy_c1:iy_c2, ix_c1:ix_c2]
    ele_coarse = (ele_coarse - ele_c) / ele_std_c
    print(ele_coarse.shape)

    std_coarse = np.loadtxt('/static_data/std26664.asc')
    std_coarse = std_coarse[iy_c1:iy_c2, ix_c1:ix_c2]
    std_coarse = (std_coarse - std_c) / std_std_c

    ele_median = np.loadtxt('/static_data/ele13332.asc')
    ele_median = ele_median[iy_m1:iy_m2,ix_m1:ix_m2]
    ele_median = (ele_median - ele_m) / ele_std_m

    std_median = np.loadtxt('/static_data/std13332.asc')
    std_median = std_median[iy_m1:iy_m2,ix_m1:ix_m2]
    std_median = (std_median - std_m) / std_std_m

    ele_fine = np.loadtxt('/static_data/ele03333.asc')
    ele_fine = ele_fine[iy_f1:iy_f2,ix_f1:ix_f2]
    ele_fine = (ele_fine - ele_f) / ele_std_f

    std_fine = np.loadtxt('/static_data/std03333.asc')
    std_fine = std_fine[iy_f1:iy_f2,ix_f1:ix_f2]
    std_fine = (std_fine - std_f) / std_std_f

    dd=ele_coarse.shape
    aa = ele_fine.shape
    topo = np.empty([aa[0], aa[1], 2])
    topo[:, :, 0] = ele_fine
    topo[:, :, 1] = std_fine
    cc = ele_median.shape
    topo1 = np.empty([cc[0], cc[1], 2])
    topo1[:, :, 0] = ele_median
    topo1[:, :, 1] = std_median

    print(topo.shape)

    # if rr==2:
    #     pre_clip=pre1[iy_c1:iy_c2, ix_c1:ix_c2,:]
    #     rain_1=np.mean(pre_clip)
    #     rain_std_1=np.std(pre_clip)
    #
    #     pre_clip = pre2[iy_m1:iy_m2, ix_m1:ix_m2, :]
    #     rain_2 = np.mean(pre_clip)
    #     rain_std_2 = np.std(pre_clip)
    # elif rr==4:
    #     pre_clip = pre1[iy_m1:iy_m2, ix_m1:ix_m2, :]
    #     rain_1 = np.mean(pre_clip)
    #     rain_std_1 = np.std(pre_clip)
    #
    #     pre_clip = pre2[iy_f1:iy_f2, ix_f1:ix_f2, :]
    #     rain_2 = np.mean(pre_clip)
    #     rain_std_2 = np.std(pre_clip)

    n_step = 365
    n_step1 = 365
    if rr==2:
        net_in = np.empty((n_step+n_step1, dd[0], dd[1], 3))
        target_in = np.empty((n_step+n_step1, cc[0], cc[1], 1))
    elif rr==4:
        net_in = np.empty((n_step+n_step1, cc[0], cc[1], 3))
        target_in = np.empty((n_step+n_step1, aa[0], aa[1], 1))
    else:
        print('The ratio of downscaling should be defined')
        sys.exit()

    ee=pre1.shape
    for i in range(0,ee[2]):
        if rr==2:
            pre_temp = pre1[iy_c1:iy_c2, ix_c1:ix_c2,i]
            pre_temp1=pre2[iy_m1:iy_m2, ix_m1:ix_m2,i]
            net_in[i, :, :, 0] = (pre_temp-rain_c)/rain_std_c
            net_in[i, :, :, 1] = ele_coarse
            net_in[i, :, :, 2] = std_coarse
            target_in[i, :, :, 0] = pre_temp1
            topodata = topo1
        elif rr==4:
            pre_temp = pre1[iy_m1:iy_m2, ix_m1:ix_m2,i]
            pre_temp1 = pre2[iy_f1:iy_f2, ix_f1:ix_f2,i]
            net_in[i, :, :, 0] = (pre_temp-rain_c)/rain_std_c
            net_in[i, :, :, 1] = ele_median
            net_in[i, :, :, 2] = std_median
            target_in[i, :, :, 0] = pre_temp1
            topodata = topo

    dd=net_in.shape
    print(dd)
    ee=target_in.shape
    print(ee)
    return topodata,net_in,target_in

def train_SRD(net_in,target_in,topo,tile,rr):

    sess = tf.InteractiveSession()
    n_batchsize = 32
    n_epoch=2000
    print_freq=10

    # define placeholder
    aa=net_in.shape
    cc=target_in.shape
    print('cc',target_in.shape)
    x = tf.placeholder(tf.float32, shape=[None, aa[1], aa[2], 3], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, cc[1], cc[2], 1], name='y_')
    # define the network
    W_init = tf.truncated_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)

    net = InputLayer(x, name='input')
    net = Conv2d(net, n_filter=64, filter_size=(5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='net1')
    #net = Conv2d(net, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='net2')
    net = Conv2d(net, n_filter=rr ** 2, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',name='net3')
    net = SubpixelConv2d(net, scale=rr, n_out_channel=1, name='subpixel_conv2d1')

    WW2 = tf.get_variable(name='WW2', shape=(3, 3, 3, 1), initializer=W_init, )
    bb2 = tf.get_variable(name='bb2', shape=(1), initializer=b_init, )
    net = myconv.mycov2D(net, topo, n_batchsize, WW2, bb2, shape=(1, 1, 2, 1), strides=(1, 1, 1, 1), padding='SAME',act=tf.nn.relu)
    net = Conv2d(net, n_filter=1, filter_size=(1, 1), strides=(1, 1),  padding='SAME', name='net4')

    y = net.outputs
    print('output:', y.shape)
    cost = tl.cost.mean_squared_error(y, y_, is_mean=True)
    cost = cost + tf.contrib.layers.l2_regularizer(0.001)(net.all_params[0])+\
           tf.contrib.layers.l2_regularizer(0.001)(net.all_params[2])+\
           tf.contrib.layers.l2_regularizer(0.001)(net.all_params[4])+\
           tf.contrib.layers.l2_regularizer(0.001)(net.all_params[6])
    #cost = relative_RMSE(y,y_)
    #cost = tl.cost.absolute_difference_error(y, y_,is_mean=True)

    train_params = net.all_params

    global_step = tf.Variable(0, trainable=False)
    learning_rate0 = tf.train.exponential_decay(0.008,
                                                global_step=global_step,
                                                decay_steps=1000,
                                                decay_rate=0.95)

    opt = tf.train.GradientDescentOptimizer(learning_rate0)
    add_global = global_step.assign_add(1)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate0).minimize(cost, var_list=train_params)
    tl.layers.initialize_global_variables(sess)
    net.print_params()
    net.print_layers()

    start_time_begin = time.time()
    for epoch in range(n_epoch):
        start_time=time.time()
        loss_ep = 0
        n_step = 0
        for X_train_a, y_train_a in iterate.minibatches(net_in, target_in, n_batchsize, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(net.all_drop)  # enable noise layers
            _, loss, add, rate = sess.run([train_op, cost, add_global, learning_rate0], feed_dict=feed_dict)
            loss_ep += loss
            n_step += 1
        loss_ep = loss_ep / n_step
        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs, loss %f, add_global %d, rate %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep, add, rate))
        if epoch + 1 == 1:
            loss_s=loss_ep
        if (epoch==30 ) and (loss_s/loss_ep < 2):
            loss_e = loss_ep
            print('The model does not converge')
            break

    loss_e=loss_ep
    net.print_params()
    model_path = '/para_dir/model_params_var_' + tile + '_' + '%s'%rr + 'r.npz'
    tl.files.save_npz(net.all_params, name=model_path)
    print("Total training time: %fs" % (time.time() - start_time_begin))
    sess.close()
    return loss_s,loss_e



rr=4     #Change here
pre1, pre2 = read_pre(rr)

tile_info=xlrd.open_workbook('/static_data/tiles_info_25deg.xls')
table = tile_info.sheet_by_name(u'tiles_info')
id_tile=table.col_values(0)
id_tile=id_tile[3:105]             #change here
#kk=3
# id_tile=id_tile[74:75]
# kk=3
# loc=[394]
train_loss=np.zeros((102,3))
for i in range(0,102):#5,6,14,74):#range(12,13):#range(12,13):#(1,11,18,24,36,38,39,43,52,53,54,55,60,62,65,69,74,81,82,87,93):#
    # tile0=int(i)
    tile='%s'%id_tile[i]
    print('start training tile %s'%i)
    # k=kk
    k=i+3
    iy_c1=int(table.cell(k,11).value-1)
    iy_c2=int(table.cell(k,12).value)
    ix_c1=int(table.cell(k,9).value-1)
    ix_c2=int(table.cell(k,10).value)
    iy_m1=int(table.cell(k,37).value-1)
    iy_m2=int(table.cell(k,38).value)
    ix_m1=int(table.cell(k,35).value-1)
    ix_m2=int(table.cell(k,36).value)
    iy_f1=int(table.cell(k,25).value-1)
    iy_f2=int(table.cell(k,26).value)
    ix_f1=int(table.cell(k,23).value-1)
    ix_f2=int(table.cell(k,24).value)
    ele_c=table.cell(k,3).value
    ele_std_c=table.cell(k,4).value
    std_c=table.cell(k,7).value
    std_std_c=table.cell(k,8).value
    ele_m=table.cell(k,29).value
    ele_std_m=table.cell(k,30).value
    std_m=table.cell(k,33).value
    std_std_m=table.cell(k,34).value
    ele_f=table.cell(k,17).value
    ele_std_f=table.cell(k,18).value
    std_f=table.cell(k,21).value
    std_std_f=table.cell(k,22).value
    rain_c=table.cell(k,13).value
    rain_std_c=table.cell(k,14).value

    #preparing input data

    topo, net_in, target_in = data_prepare(iy_c1, iy_c2, ix_c1, ix_c2,
                                           iy_m1, iy_m2, ix_m1, ix_m2,
                                           iy_f1, iy_f2, ix_f1, ix_f2,
                                           ele_c, ele_std_c,
                                           std_c, std_std_c,
                                           ele_m, ele_std_m,
                                           std_m, std_std_m,
                                           ele_f, ele_std_f,
                                           std_f, std_std_f,
                                           rain_c, rain_std_c,
                                           tile, rr, pre1, pre2)
    #training model
    tf.reset_default_graph()
    loss_s,loss_e=train_SRD(net_in, target_in, topo, tile,rr)
    train_loss[i,0]=i
    train_loss[i, 1] = loss_s
    train_loss[i, 2] = loss_e
    np.savetxt('train_loss.txt', train_loss, fmt='%7.2f')
    # kk=kk+1
