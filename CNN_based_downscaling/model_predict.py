import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.files import *
import mycov2D
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

def tif_info(in_path):
    dataset = gdal.Open(in_path)
    if dataset == None:
        print(in_path + " cannot be opened")
        return
    width = dataset.RasterXSize  # 栅格矩阵的列数
    height = dataset.RasterYSize  # 栅格矩阵的行数
    bands = dataset.RasterCount  # 波段数
    proj=dataset.GetProjection()
    gts = dataset.GetGeoTransform()
    return width,height,bands,proj,gts

def read_pre(samples,ynum,rr):
    n_step=samples
    date1 = datetime.datetime(ynum, 1, 1, 0, 0, 0) ####Change here
    t_step = 0
    while t_step < n_step:
        date2 = date1 + datetime.timedelta(days=t_step)
        t_str = str(date2.strftime('%Y%m%d'))
        if rr==2:
            path='/ERA5/' + '%s/'%ynum + t_str + '.tif'
        elif rr==4:
            path = '/ERA5_CNN/var/' + 'out_2r/' + '%s/' % ynum + t_str + '.tif'
        else:
            print('The ratio of downscaling should be defined')
            sys.exit()
        print(path)
        pre0 = read_tif(path)
        pre0 = np.reshape(pre0, (pre0.shape[0], pre0.shape[1], 1))
        if t_step==0:
            pre=pre0
        else:
            pre=np.concatenate((pre,pre0),axis=2)
        t_step=t_step+1

    return pre


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
                 tile,rr,ynum,samples,pre_in):
    ################################
    ####### data processing

    # load topography data
    ele_coarse = np.loadtxt('/static_data/ele26664.asc')
    # ele_coarse=(ele_coarse-2442)/1807
    ele_coarse = ele_coarse[iy_c1:iy_c2, ix_c1:ix_c2]
    ele_coarse = (ele_coarse - ele_c) / ele_std_c
    std_coarse = np.loadtxt('/static_data/std26664.asc')
    #std_coarse = (std_coarse - 231) / 222
    std_coarse = std_coarse[iy_c1:iy_c2, ix_c1:ix_c2]
    std_coarse = (std_coarse - std_c) / std_std_c

    ele_median = np.loadtxt('/static_data/ele13332.asc')
    #ele_median = (ele_median - 2442) / 1817
    ele_median = ele_median[iy_m1:iy_m2,ix_m1:ix_m2]
    ele_median = (ele_median - ele_m) / ele_std_m
    std_median = np.loadtxt('/static_data/std13332.asc')
    #std_median = (std_median - 178) / 186
    std_median = std_median[iy_m1:iy_m2,ix_m1:ix_m2]
    std_median = (std_median - std_m) / std_std_m

    ele_fine = np.loadtxt('/static_data/ele03333.asc')
    #ele_fine = (ele_fine - 2422) / 1829
    ele_fine = ele_fine[iy_f1:iy_f2,ix_f1:ix_f2]
    ele_fine = (ele_fine - ele_f) / ele_std_f
    std_fine = np.loadtxt('/static_data/std03333.asc')
    #std_fine = (std_fine - 94) / 109
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
    date1 = datetime.datetime(ynum, 1, 1, 0, 0, 0, 0)              #change here
    n_step = samples    #change here
    if rr==2:
        net_in = np.empty((n_step, dd[0], dd[1], 3))
        topodata = topo1
    elif rr==4:
        net_in = np.empty((n_step, cc[0], cc[1], 3))
        topodata = topo
    else:
        print('The ratio of downscaling should be defined')
        sys.exit()

    t_step = 0
    while t_step < n_step:
        date2 = date1 + datetime.timedelta(days=t_step)
        t_str = str(date2.strftime('%Y%m%d'))

        if rr==2:
            
            pre_temp=pre_in[:,:,t_step]
            pre_temp = pre_temp[iy_c1:iy_c2, ix_c1:ix_c2]
            #print(pre_temp.shape)
            pre_temp = (pre_temp - rain_c) / rain_std_c
            net_in[t_step, :, :, 0] = pre_temp
            net_in[t_step, :, :, 1] = ele_coarse
            net_in[t_step, :, :, 2] = std_coarse
        elif rr==4:
           
            pre_temp=pre_in[:,:,t_step]
            pre_temp = pre_temp[iy_m1:iy_m2, ix_m1:ix_m2]
            # print(pre_temp.shape)
            pre_temp = (pre_temp - rain_c) / rain_std_c
            net_in[t_step, :, :, 0] = pre_temp
            net_in[t_step, :, :, 1] = ele_median
            net_in[t_step, :, :, 2] = std_median
        else:
            print('The ratio of downscaling should be defined')
            sys.exit()
        t_step = t_step + 1
    dd=net_in.shape
    print(dd)
    return (topodata,net_in)

def predict_SRD(net_in,topo,tile,rr,ynum,samples):
    sess = tf.InteractiveSession()
    n_batchsize = samples       #change here

    # define placeholder
    aa = net_in.shape
    print('Inputs shape:', aa)
    x = tf.placeholder(tf.float32, shape=[None, aa[1], aa[2], 3], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, rr*aa[1], rr*aa[2], 1], name='y_')
    # define the network
    W_init = tf.truncated_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)

    net = InputLayer(x, name='input')
    net = Conv2d(net, n_filter=64, filter_size=(5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='net1')
    #net = Conv2d(net, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='net2')
    net = Conv2d(net, n_filter=rr ** 2, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='net3')
    net = SubpixelConv2d(net, scale=rr, n_out_channel=1, name='subpixel_conv2d1')

    WW2 = tf.get_variable(name='WW2', shape=(3, 3, 3, 1), initializer=W_init, )
    bb2 = tf.get_variable(name='bb2', shape=(1), initializer=b_init, )
    net = mycov2D.mycov2D1(net, topo, n_batchsize, WW2, bb2, shape=(1, 1, 2, 1), strides=(1, 1, 1, 1), padding='SAME',
                           act=tf.nn.relu)
    net = Conv2d(net, n_filter=1, filter_size=(1, 1), strides=(1, 1), padding='SAME', name='net4')

    y = net.outputs
    y_op = tf.identity(y)
    model_path='model_params_var' + tile + '_' + '%s'%rr + 'r.npz'
    # params = load_npz(path='D:/study/Rdownscale/code/SRD_downscale/venv/Scripts/', name=model_path)
    params = load_npz(path='/para_dir/', name=model_path)
    assign_params(sess, params, net)
    print(net.all_params)
    rain_predict = tl.utils.predict(sess, net, net_in, x, y_op, batch_size=None)
    rain_predict = np.squeeze(rain_predict)
    print(rain_predict.shape)
    sess.close()
    # t_start = datetime.datetime(ynum, 1, 1, 0, 0, 0, 0)             #change here
    # for i in range(0, samples):           #change here
    #     t_current = t_start + datetime.timedelta(days=i)
    #     t_str = str(t_current.strftime('%Y%m%d'))
    #     out_dir1 = 'F:/Rdownscale/data/ERA5_CNN/prcp/' + '%s'%ynum + '/' + tile + '/'     #change here
    #     if not os.path.exists(out_dir1):
    #         os.mkdir(out_dir1)
    #     out_dir2=out_dir1 + 'out_' + '%s'%rr + 'r/'
    #     if not os.path.exists(out_dir2):
    #         os.mkdir(out_dir2)
    #     path_out=out_dir2 + t_str + '.txt'
    #
    #     print(path_out)
    #     rain_d = rain_predict[i, :, :]
    #     rain_d[rain_d < 0] = 0
    #     np.savetxt(path_out, rain_d, fmt='%7.2f')
    return rain_predict


def main_function(yy_in):
    if calendar.isleap(yy_in):
        n_sample=366
    else:
        n_sample=365
    # n_sample = 2  ############################## CHANGE HERE ##############################################
    tile_info = xlrd.open_workbook('/static_data/tiles_info_25deg.xls')
    table = tile_info.sheet_by_name(u'tiles_info')
    id_tile = table.col_values(0)
    id_tile = id_tile[3:105]  # change here for different tiles seperation
    kk = 3
    rr = 2  # Change here

    pre_in=read_pre(n_sample,yy_in,rr)
    if rr==2:
        pre_out = np.zeros((n_sample,118,336))
        rr1 = 2
    elif rr==4:
        pre_out = np.zeros((n_sample, 472, 1344))
        rr1 = 8

    print(pre_out.shape)


    #######################################################################

    for i in range(1, 7):
        for j in range(1, 18):
            tile = str(i).zfill(2) + str(j).zfill(2)
            print('start correct tile %s' % tile)
            loc = id_tile.index(tile)
            iy_c1 = int(table.cell(loc+3, 11).value - 1)
            iy_c2 = int(table.cell(loc+3, 12).value)
            ix_c1 = int(table.cell(loc+3, 9).value - 1)
            ix_c2 = int(table.cell(loc+3, 10).value)
            iy_m1 = int(table.cell(loc+3, 37).value - 1)
            iy_m2 = int(table.cell(loc+3, 38).value)
            ix_m1 = int(table.cell(loc+3, 35).value - 1)
            ix_m2 = int(table.cell(loc+3, 36).value)
            iy_f1 = int(table.cell(loc+3, 25).value - 1)
            iy_f2 = int(table.cell(loc+3, 26).value)
            ix_f1 = int(table.cell(loc+3, 23).value - 1)
            ix_f2 = int(table.cell(loc+3, 24).value)
            ele_c = table.cell(loc+3, 3).value
            ele_std_c = table.cell(loc+3, 4).value
            std_c = table.cell(loc+3, 7).value
            std_std_c = table.cell(loc+3, 8).value
            ele_m = table.cell(loc+3, 29).value
            ele_std_m = table.cell(loc+3, 30).value
            std_m = table.cell(loc+3, 33).value
            std_std_m = table.cell(loc+3, 34).value
            ele_f = table.cell(loc+3, 17).value
            ele_std_f = table.cell(loc+3, 18).value
            std_f = table.cell(loc+3, 21).value
            std_std_f = table.cell(loc+3, 22).value
            rain_c = table.cell(loc+3, 13).value
            rain_std_c = table.cell(loc+3, 14).value
            topo, net_in = data_prepare(iy_c1, iy_c2, ix_c1, ix_c2,
                                        iy_m1, iy_m2, ix_m1, ix_m2,
                                        iy_f1, iy_f2, ix_f1, ix_f2,
                                        ele_c, ele_std_c,
                                        std_c, std_std_c,
                                        ele_m, ele_std_m,
                                        std_m, std_std_m,
                                        ele_f, ele_std_f,
                                        std_f, std_std_f,
                                        rain_c, rain_std_c,
                                        tile, rr, yy_in, n_sample, pre_in)
            # predict
            tf.reset_default_graph()
            rain_predict = predict_SRD(net_in, topo, tile, rr, yy_in, n_sample)

            if rr==2:
                ix1=ix_m1
                ix2=ix_m2
                iy1=iy_m1
                iy2=iy_m2
            elif rr==4:
                ix1 = ix_f1
                ix2 = ix_f2
                iy1 = iy_f1
                iy2 = iy_f2
            print(iy1,iy2,ix1,ix2)

            if i==6:
                if j==17:
                    pre_out[:,iy1:iy2,ix1:ix2]=rain_predict
                elif j==1:
                    pre_temp1=pre_out[:,iy1:iy2,ix1:(ix1+1*rr1)]
                    pre_temp2=rain_predict[:,:,0:1*rr1]
                    pre_out[:,iy1:iy2,ix1:ix2]=rain_predict
                    pre_out[:, iy1:iy2, ix1:(ix1 + 1 * rr1)]=(pre_temp1+pre_temp2)/2
                else:
                    pre_temp1 = pre_out[:, iy1:iy2, ix1:(ix1 + 1 * rr1)]
                    pre_temp2 = rain_predict[:, :, 0:1 * rr1]
                    pre_out[:, iy1:iy2, ix1:ix2] = rain_predict
                    pre_out[:, iy1:iy2, ix1:(ix1 + 1 * rr1)] = (pre_temp1 + pre_temp2) / 2
            elif i==1:
                if j==17:
                    pre_temp1=pre_out[:,iy1:(iy1+1*rr1),ix1:ix2]
                    pre_temp2=rain_predict[:,0:1*rr1,:]
                    pre_out[:,iy1:iy2,ix1:ix2]=rain_predict
                    pre_out[:, iy1:(iy1 + 1 * rr1), ix1:ix2]=(pre_temp1+pre_temp2)/2
                elif j==1:
                    pre_temp1=pre_out[:,iy1:iy2,ix1:(ix1+1*rr1)]
                    pre_temp2=pre_out[:, iy1:(iy1 + 1 * rr1), ix1:ix2]
                    pre_temp3 = rain_predict[:, :, 0:1 * rr1]
                    pre_temp4 = rain_predict[:, 0:1 * rr1, :]
                    pre_out[:,iy1:iy2,ix1:ix2]=rain_predict
                    pre_out[:, iy1:iy2, ix1:(ix1 + 1 * rr1)]=(pre_temp1+pre_temp3)/2
                    pre_out[:, iy1:(iy1 + 1 * rr1), ix1:ix2] = (pre_temp2 + pre_temp4) / 2
                else:
                    pre_temp1 = pre_out[:, iy1:iy2, ix1:(ix1 + 1 * rr1)]
                    pre_temp2 = pre_out[:, iy1:(iy1 + 1 * rr1), ix1:ix2]
                    pre_temp3 = rain_predict[:, :, 0:1 * rr1]
                    pre_temp4 = rain_predict[:, 0:1 * rr1, :]
                    pre_out[:, iy1:iy2, ix1:ix2] = rain_predict
                    pre_out[:, iy1:iy2, ix1:(ix1 + 1 * rr1)] = (pre_temp1 + pre_temp3) / 2
                    pre_out[:, iy1:(iy1 + 1 * rr1), ix1:ix2] = (pre_temp2 + pre_temp4) / 2
            else:
                if j==17:
                    pre_temp1=pre_out[:,iy1:(iy1+1*rr1),ix1:ix2]
                    pre_temp2=rain_predict[:,0:1*rr1,:]
                    pre_out[:,iy1:iy2,ix1:ix2]=rain_predict
                    pre_out[:, iy1:(iy1 + 1 * rr1), ix1:ix2]=(pre_temp1+pre_temp2)/2
                elif j==1:
                    pre_temp1=pre_out[:,iy1:iy2,ix1:(ix1+1*rr1)]
                    pre_temp2=pre_out[:, iy1:(iy1 + 1 * rr1), ix1:ix2]
                    pre_temp3 = rain_predict[:, :, 0:1 * rr1]
                    pre_temp4 = rain_predict[:, 0:1 * rr1, :]
                    pre_out[:,iy1:iy2,ix1:ix2]=rain_predict
                    pre_out[:, iy1:iy2, ix1:(ix1 + 1 * rr1)]=(pre_temp1+pre_temp3)/2
                    pre_out[:, iy1:(iy1 + 1 * rr1), ix1:ix2] = (pre_temp2 + pre_temp4) / 2
                else:
                    pre_temp1 = pre_out[:, iy1:iy2, ix1:(ix1 + 1 * rr1)]
                    pre_temp2 = pre_out[:, iy1:(iy1 + 1 * rr1), ix1:ix2]
                    pre_temp3 = rain_predict[:, :, 0:1 * rr1]
                    pre_temp4 = rain_predict[:, 0:1 * rr1, :]
                    pre_out[:, iy1:iy2, ix1:ix2] = rain_predict
                    pre_out[:, iy1:iy2, ix1:(ix1 + 1 * rr1)] = (pre_temp1 + pre_temp3) / 2
                    pre_out[:, iy1:(iy1 + 1 * rr1), ix1:ix2] = (pre_temp2 + pre_temp4) / 2


    ##############################################################################################################


    ####输出降尺度结果
    if rr==2:
        wd, ht, bd, prj, gts = tif_info('ref_map.tif')
    elif rr==4:
        wd, ht, bd, prj, gts = tif_info('ref_map.tif')

    t_start = datetime.datetime(yy_in, 1, 1, 0, 0, 0, 0)             #change here
    for i in range(0, n_sample):           #change here
        t_current = t_start + datetime.timedelta(days=i)
        t_str = str(t_current.strftime('%Y%m%d'))
        out_dir1 = '/ERA5_CNN/var/' + 'out_' + '%s' % rr +'r/'  # change here
        if not os.path.exists(out_dir1):
            os.mkdir(out_dir1)
        out_dir2 = out_dir1 + '%s/' % yy_in
        if not os.path.exists(out_dir2):
            os.mkdir(out_dir2)
        path_out = out_dir2 + t_str + '.tif'
        print(path_out)
        rain_d = pre_out[i, :, :]
        print(rain_d.shape)
        rain_d[rain_d < 0] = 0
        # np.savetxt(path_out, rain_d, fmt='%7.2f')
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path_out, wd, ht, bd, gdal.GDT_Float32)
        dataset.SetGeoTransform(gts)
        dataset.SetProjection(prj)
        dataset.GetRasterBand(1).WriteArray(rain_d)
        del dataset
    return pre_out


for i in range(1981,2019):
    rain1,rain2=main_function(i)