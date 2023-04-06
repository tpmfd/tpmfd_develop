import pymysql
import numpy as np
import datetime as dtt
import calendar
import gdal
from bisect import bisect_left
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from joblib import parallel_backend
from joblib import dump, load
import lightgbm as lgb
from multiprocessing import Pool


def tif_info(in_path):
    dataset = gdal.Open(in_path)
    if dataset == None:
        print(in_path + " cannot be opened")
        return
    width = dataset.RasterXSize  
    height = dataset.RasterYSize  
    bands = dataset.RasterCount  
    proj = dataset.GetProjection()
    gts = dataset.GetGeoTransform()
    return width, height, bands, proj, gts


def read_tif(in_path):
    dataset = gdal.Open(in_path)
    if dataset == None:
        print(in_path + " cannot be opened")
        return
    width = dataset.RasterXSize  
    height = dataset.RasterYSize  
    bands = dataset.RasterCount  
    data = dataset.ReadAsArray(0, 0, width, height)  
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

def predict_d(i):
    dem = np.loadtxt('/static_data/ele03333.asc')
    dem_std = np.loadtxt('/static_data/std03333.asc')
    cluster = read_tif('/static_data/t_clusters.tif')
    wd, ht, bd, prj, gts = tif_info('/ERA5_CNN/var/monthly/201801.tif')

    model = load('/parms/var/final/direct_d_reg_' + '%02d' % i)
    
    ###############################################
    for j in range(1979, 2021):
        nday = calendar.monthrange(j, i)
        for jj in range(1, nday[1] + 1):
            path1 = '/ERA5_CNN/var/daily/' + '%04d' % j + '/' + '%04d' % j + '%02d' % i + '%02d' % jj + '.tif'
            var_m = read_tif(path1)
            #############################################
            var_m[0, :] = var_m[3, :]
            var_m[1, :] = var_m[3, :]
            var_m[2, :] = var_m[3, :]
            var_m[:, 0] = var_m[:, 5]
            var_m[:, 1] = var_m[:, 5]
            var_m[:, 2] = var_m[:, 5]
            var_m[:, 3] = var_m[:, 5]
            var_m[:, 4] = var_m[:, 5]
            ############################################
            dims = var_m.shape
            var_m = np.concatenate((np.reshape(var_m[:, 0], (dims[0], 1)), var_m), axis=1)
            var_m = np.concatenate((var_m, np.reshape(var_m[:, dims[1]], (dims[0], 1))), axis=1)
            var_m = np.concatenate((np.reshape(var_m[0, :], (1, dims[1] + 2)), var_m), axis=0)
            var_m = np.concatenate((var_m, np.reshape(var_m[dims[0], :], (1, dims[1] + 2))), axis=0)

            ratio_m = var_m 
            ###################################

            feature = np.zeros((472 * 1344, 14))
            k = 0
            for iy in range(0, 472):
                for ix in range(0, 1344):
                    feature[k, 0] = dem[iy, ix]
                    feature[k, 1] = dem_std[iy, ix]
                    feature[k, 2] = cluster[iy, ix]

                    ix1 = ix + 1
                    iy1 = iy + 1

                    var_temp = ratio_m[iy1 - 1:iy1 + 2, ix1 - 1:ix1 + 2]
                    var_temp = np.reshape(var_temp, (1, 9))
                    feature[k, 3:12] = var_temp
                    feature[k, 12] = 60.86668 + 0.03333 / 2 + ix * 0.03333
                    feature[k, 13] = 41.46509 - 0.03333 / 2 - iy * 0.03333
                    # feature[k, 14] = j

                    k = k + 1


            ##############################################################################

            predict = model.predict(feature)
            
            ##############################################################################

            var_c = np.reshape(predict, (472, 1344))

            ############################
            path3 = '/data/var/daily_4s/RF_direct_val/' + '%04d/' % j +'%04d' % j + '%02d' % i + '%02d' % jj+ '.tif'
            print(path3)
            driver = gdal.GetDriverByName("GTiff")
            dataset = driver.Create(path3, wd, ht, bd, gdal.GDT_Float32)
            dataset.SetGeoTransform(gts)
            dataset.SetProjection(prj)
            dataset.GetRasterBand(1).WriteArray(var_c)
            del dataset


if __name__ == '__main__':
    p=Pool(12) 
    for i in range(1,13): 
        r=p.apply_async(predict_d,args=(i,)) 
    p.close() 
    p.join()  



