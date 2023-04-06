import pymysql
import numpy as np
import datetime as dtt
import gdal
from bisect import bisect_left
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
from joblib import parallel_backend
from joblib import dump, load

from scipy import spatial
from pykrige.ok import OrdinaryKriging
from multiprocessing import Pool



def tif_info(in_path):
    dataset = gdal.Open(in_path)
    if dataset == None:
        print(in_path + " cannot be opened")
        return
    width = dataset.RasterXSize  
    height = dataset.RasterYSize  
    bands = dataset.RasterCount  
    proj=dataset.GetProjection()
    gts = dataset.GetGeoTransform()
    return width,height,bands,proj,gts

def read_tif(in_path):
    dataset = gdal.Open(in_path)
    if dataset == None:
        print(in_path + " cannot be opened")
        return
    width = dataset.RasterXSize  
    height = dataset.RasterYSize  
    bands = dataset.RasterCount 
    data = dataset.ReadAsArray(0, 0, width, height)  # 获取数据
    return data

def loc_find(lon,lat):
    ds = gdal.Open('/ERA5_CNN/var/monthly/202001.tif')
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    lon0=[]
    lat0=[]
    for i in range(0,width+1):
        lon0.append(gt[0]+i*0.03333)
    for i in range(0,height+1):
        lat0.append(gt[3]-i*0.03333)

    lon0=np.array(lon0)
    lat0=np.array(lat0)
    for i in range(1,len(lon0)):
        if (lon>=lon0[i-1]) and (lon<lon0[i]):
            loc_lon=i-1
            lon_g=lon0[i-1]+0.03333/2
    for i in range(1,len(lat0)):
        if (lat<=lat0[i-1]) and (lat>lat0[i]):
            loc_lat=i-1
            lat_g=lat0[i]+0.03333/2
    return loc_lat,loc_lon,lon_g,lat_g



def krig_d(t_step):
    dem=np.loadtxt('/static_data/ele03333.asc')
    conn=pymysql.connect(host='localhost',port=3306,user='mysqlid',passwd='mypassword',database='meteo_obs')
    cur=conn.cursor()
    wd, ht, bd, prj, gts = tif_info('/ERA5_CNN/var/monthly/201801.tif')

    t_start=dtt.date(1979,1,1)

    t_current=t_start+dtt.timedelta(days=t_step)
    t_str0=str(t_current.strftime('%Y'))
    t_str1=str(t_current.strftime('%Y%m%d'))
    path1='data/var/daily_4s/RF_direct_val/'+t_str0+'/'+t_str1+'.tif'
    print(path1)
    var_m=read_tif(path1)
    

    # print(t_current)
    sql = "SELECT ID,var FROM daily_var WHERE date_utc = '%s'" % t_current
    cur.execute(sql)
    value_select = cur.fetchall()
    value_select = np.array(value_select)
    id_sta = value_select[:, 0]
    var_o = value_select[:, 1].astype(np.float)

    #############################################################################

    resV_temp=np.zeros((len(id_sta)))
    cord0=np.zeros((len(id_sta),2))
    

    for j in range(0,len(id_sta)):
        sql1 = "SELECT Lon,Lat,Ele FROM sta_var WHERE ID = '%s'" % (id_sta[j])
        cur.execute(sql1)
        sta_info = cur.fetchall()
        sta_info = np.array(sta_info)
        sta_info = sta_info.astype(np.float)
        loc_1, loc_2,lon_g,lat_g = loc_find(sta_info[0,0], sta_info[0,1])
        # var_o[j]=var_o[j]-0.6*(dem[loc_1,loc_2]-sta_info[0,2])/100  #for temp
        # var_o[j]=var_o[j]-(dem[loc_1, loc_2]-sta_info[0,2])/9     for pres
        resV_temp[j]=var_o[j]-var_m[loc_1,loc_2]
        cord0[j,0]=lon_g
        cord0[j,1]=lat_g

    cord1,idx_0=np.unique(cord0,axis=0,return_inverse=True)
    resV=np.zeros((cord1.shape[0]))
    for j in range(0,cord1.shape[0]):
        resV[j]=np.mean(resV_temp[idx_0==j])

    ###########################################################################################################
    krig = OrdinaryKriging(cord1[:,0],cord1[:,1],resV,variogram_model='spherical',coordinates_type='geographic')
    
    ###########################################################################################################
    gridx=np.zeros((1344))
    for j in range(0,1344):
        gridx[j]=60.86668+j*0.03333

    gridy=np.zeros((472))
    for j in range(0,472):
        gridy[j]=41.46509-j*0.03333

    resE=np.zeros((472,1344))
    resE[0:100,:],ssE=krig.execute("grid", gridx, gridy[0:100])
    print('line 100')
    resE[100:200,:],ssE=krig.execute("grid", gridx, gridy[100:200])
    print('line 200')
    resE[200:300,:],ssE=krig.execute("grid", gridx, gridy[200:300])
    print('line 300')
    resE[300:400,:],ssE=krig.execute("grid", gridx, gridy[300:400])
    print('line 400')
    resE[400:472,:],ssE=krig.execute("grid", gridx, gridy[400:472])

    #############################################################################################################

    var_c=var_m+resE

    path2='/data/var/daily_4s/krg_direct_val/'+t_str0+'/'+t_str1+'.tif' 
    print(path2)

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path2, wd, ht, bd, gdal.GDT_Float32)
    dataset.SetGeoTransform(gts)
    dataset.SetProjection(prj)
    dataset.GetRasterBand(1).WriteArray(var_c)
    del dataset


    cur.close()
    conn.close()

    return


if __name__ == '__main__':
    p=Pool(30) 
    for i in range(0,15372): #
        r=p.apply_async(krig_d,args=(i,)) 
    p.close() #关闭进程池
    p.join()  #结束



