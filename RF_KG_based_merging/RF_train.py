import pymysql
import pandas as pd
import numpy as np
import datetime as dtt
import calendar
import gdal
from bisect import bisect_left
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from joblib import parallel_backend
from joblib import dump, load
import lightgbm as lgb

from sklearn import preprocessing
from sklearn.impute import SimpleImputer


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


def loc_find(lon, lat):
    ds = gdal.Open('/ERA5_CNN/var/monthly/202001.tif')
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    lon0 = []
    lat0 = []
    for i in range(0, width + 1):
        lon0.append(gt[0] + i * 0.03333)
    for i in range(0, height + 1):
        lat0.append(gt[3] - i * 0.03333)

    lon0 = np.array(lon0)
    lat0 = np.array(lat0)
    for i in range(1, len(lon0)):
        if (lon >= lon0[i - 1]) and (lon < lon0[i]):
            loc_lon = i - 1
            lon_g = lon0[i - 1] + 0.03333 / 2
    for i in range(1, len(lat0)):
        if (lat <= lat0[i - 1]) and (lat > lat0[i]):
            loc_lat = i - 1
            lat_g = lat0[i] + 0.03333 / 2
    return loc_lat, loc_lon, lon_g, lat_g


dem = np.loadtxt('/static_data/ele03333.asc')
dem_std = np.loadtxt('/static_data/std03333.asc')
cluster = read_tif('/static_data/t_clusters.tif')

conn = pymysql.connect(host='localhost', port=3306, user='mysqlid', passwd='mypassword', database='meteo_obs')
cur = conn.cursor()



sql1 = "SELECT ID,Lon,Lat,Ele FROM sta_var "
cur.execute(sql1)
sta_info0 = cur.fetchall()
sta_info0 = np.array(sta_info0)
id0=sta_info0[:,0]
id0=id0.tolist()
sta_info1 = sta_info0[:,1:4].astype(np.float)

cor0=[]
for i in range(1,13):
    print(i)
    feature0 = np.zeros((11282880, 14))
    target0 = np.zeros((11282880))
    id_sta1= []
    kkk=0

    for j in range(1979, 2021):
        nday = calendar.monthrange(j, i)
        current_date1 = dtt.date(j, i, 1)
        current_date2 = dtt.date(j, i, nday[1])
        sql = "SELECT ID,date_utc,var FROM daily_var WHERE date_utc between '%s' and '%s'" % (current_date1,current_date2)
        cur.execute(sql)
        value_select = cur.fetchall()
        value_select = np.array(value_select)
        id_sta0 = value_select[:, 0]
        date_s=value_select[:, 1]
        var_o_m00 = value_select[:, 2].astype(np.float)

        for jj in range(1,nday[1]+1):
            path2 = '/ERA5_CNN/var/daily/' + '%04d' % j +'/'+ '%04d' % j+ '%02d' % i + '%02d' % jj +'.tif'
            print(path2)
            var_m0 = read_tif(path2)
            #############################################
            var_m0[0, :] = var_m0[3, :]
            var_m0[1, :] = var_m0[3, :]
            var_m0[2, :] = var_m0[3, :]
            var_m0[:, 0] = var_m0[:, 5]
            var_m0[:, 1] = var_m0[:, 5]
            var_m0[:, 2] = var_m0[:, 5]
            var_m0[:, 3] = var_m0[:, 5]
            var_m0[:, 4] = var_m0[:, 5]
            # ############################################
            dims = var_m0.shape
            var_m0 = np.concatenate((np.reshape(var_m0[:, 0], (dims[0], 1)), var_m0), axis=1)
            var_m0 = np.concatenate((var_m0, np.reshape(var_m0[:, dims[1]], (dims[0], 1))), axis=1)
            var_m0 = np.concatenate((np.reshape(var_m0[0, :], (1, dims[1] + 2)), var_m0), axis=0)
            var_m0 = np.concatenate((var_m0, np.reshape(var_m0[dims[0], :], (1, dims[1] + 2))), axis=0)

            ratio_m = var_m0
            # ###########################################
            current_date = dtt.date(j, i, jj)
            id_sta=id_sta0[date_s==current_date]
            var_o_m0=var_o_m00[date_s==current_date]


            for k in range(0, len(id_sta)):

                loc_0=id0.index(id_sta[k])
                loc_1, loc_2, lon_g, lat_g = loc_find(sta_info1[loc_0,0], sta_info1[loc_0,1])
                ##################################
                # var_o_m0[k]=var_o_m0[k]-0.6*(dem[loc_1,loc_2]-sta_info1[loc_0,2])/100  #for temp
                # var_o_m0[k]=var_o_m0[k]-(dem[loc_1, loc_2]-sta_info1[loc_0,2])/9    #for pres

                target0[kkk] = var_o_m0[k]
                feature0[kkk, 0] = dem[loc_1, loc_2]
                feature0[kkk, 1] = dem_std[loc_1, loc_2]
                feature0[kkk, 2] = cluster[loc_1, loc_2]

                loc_11 = loc_1 + 1  # 由于在原数据周边增加了一圈，相应的索引加1
                loc_22 = loc_2 + 1
                var_temp = ratio_m[loc_11 - 1:loc_11 + 2, loc_22 - 1:loc_22 + 2]
                var_temp = np.reshape(var_temp, (1, 9))
                feature0[kkk, 3:12] = var_temp
                feature0[kkk, 12] = lon_g
                feature0[kkk, 13] = lat_g
                id_sta1.append(id_sta[k])
                kkk=kkk+1





    id_sta1=np.array(id_sta1)
    feature=feature0[0:len(id_sta1),:]
    target=target0[0:len(id_sta1)]

    feature_bck = 1 * feature

    feature = feature[~np.isnan(feature_bck).any(axis=1), :]
    target = target[~np.isnan(feature_bck).any(axis=1)]
    id_sta1 = id_sta1[~np.isnan(feature_bck).any(axis=1)]

    target_bck = target * 1
    feature = feature[~np.isnan(target_bck), :]
    target = target[~np.isnan(target_bck)]
    id_sta1 = id_sta1[~np.isnan(target_bck)]

    ####################################
    f0 = open('/train_in/var_d_all_direct_' + '%02d' % i + '.txt','w')
    for j in range(0, len(target)):
        out_str = '%17s' % id_sta1[j] + '%8.2f' % feature[j,0] + '%8.2f' % feature[j,1] + '%3d' % feature[j,2]
        f0.write(out_str)
        for jj in range(3,12):
            out_str = '%8.2f' % feature[j, jj]
            f0.write(out_str)
        out_str = '%9.3f' % feature[j, 12]+'%9.3f' % feature[j, 13]+'%8.2f' % target[j]+'\n'
        f0.write(out_str)

    f0.close()
    ######################################
    # data=np.loadtxt('/train_in/var_d_all_direct_'+'%02d'%i+'.txt',dtype=str)
    # id_sta1=data[:,0]
    # feature=data[:,1:15]
    # target=data[:,15]
    # feature=feature.astype(np.float)
    # target=target.astype(np.float)
    ###########################################
    print(feature.shape[0])



    model = RandomForestRegressor(n_estimators=120, max_features=0.1, min_samples_leaf=10,max_samples=0.8,
                                  n_jobs=5,
                                  verbose=1)
    model.fit(feature, target)

    model_file = '/parms/var/final/direct_d_reg_' + '%02d' % i
    dump(model, model_file)

cur.close()
conn.close()
