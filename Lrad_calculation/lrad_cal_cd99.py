import numpy as np
import datetime as dtt
import calendar
import time
import gdal
from bisect import bisect_left
import netCDF4 as nc
from multiprocessing import Pool
import math



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
    data = dataset.ReadAsArray(0, 0, width, height)  
    return data

def nc_read(in_path,var):
	f=nc.Dataset(in_path,'r')
	da=f.variables[var]
	da=da[:].data
	return da


def nc_write(in_path,data,nhour):
	lons=np.zeros((1339))
	for i in range(0,1339):
		lons[i]=60.86668+0.03333/2+5*0.03333+i*0.03333
	lats=np.zeros((469))
	for i in range(0,469):
		lats[i]=41.46509-0.03333/2-3*0.03333-i*0.03333

	f=nc.Dataset(in_path, 'w',format="NETCDF4")
	f.description='High-resolution surface downward longwave radiation for the Third Pole region'
	f.history = 'Created at ' + time.ctime(time.time())
	f.institution='Department of Earth System Science, Tsinghua University'

	f.createDimension('time',24)
	f.createDimension('latitude',len(lats))
	f.createDimension('longitude',len(lons))
	times = f.createVariable('time', 'f8', ('time',))
	lat = f.createVariable('latitude', 'f8', ('latitude',))
	lon = f.createVariable('longitude', 'f8', ('longitude',))
	var = f.createVariable('lrad','i4',('time','latitude','longitude'),zlib=True)
	# var = f.createVariable('prcp','f8',('time','latitude','longitude'))
	times.units='hours since 1900-1-1 00:00:00'
	lat.units = 'degrees north'
	lon.units = 'degrees east'
	var.units='W m-2'
	var.longname='surface downward longwave radiation'
	var.setncattr('scale_factor',0.01)
	var.setncattr('add_offset',0.0)
	times[:]=nhour
	lat[:] = lats[:]
	lon[:] = lons[:]
	var[:] = data
	f.close()
	return

def t2es(tem):
	tem=tem-273.15
	es=6.11*10**(7.5*tem/(237.3+tem))
	es[tem<0.0]=6.11*10**(9.5*tem[tem<0.0]/(265.5+tem[tem<0.0]))
	return es


def solarR0(lon0, lon, lat, yy, mm, dd, hh, mn, ss):
	# print('calculating the solarR0................')
	# !---------------------------------------------------------------------------------------------------
	# !
	# !     PURPOSE:
	# !
	# !     Calculate horizontal extraterrestial solar insolation (w/s).
	# ! 
	# !---------------------------------------------------------------------------------------------------

	I00 = 1353
	# hsun=np.zeros((469,1339))

	t_deta=dtt.datetime(yy, mm, dd, hh, mn, ss)-dtt.datetime(yy, 1, 1, 0, 0, 0)
	jday=t_deta.days+1

	t_deta=dtt.datetime(yy, 12, 31, 0, 0, 0)-dtt.datetime(yy, 1, 1, 0, 0, 0)
	jday0=t_deta.days+1

	w = 2 * np.pi * jday / jday0
	d0d2 = 1.00011 + 0.034221 * np.cos(w) + 0.00128 * np.sin(w) + 0.000719 * np.cos(2 * w) + 0.000077 * np.sin(2 * w)
	delta = 0.3622133 - 23.24763 * np.cos(w + 0.153231) - 0.3368908 * np.cos(2.0 * w + 0.2070988) - 0.1852646 * np.cos(3.0 * w + 0.6201293)
	delta = delta * np.pi / 180
	eta = 60.0 * (-0.0002786409 + 0.1227715 * np.cos(w + 1.498311) - 0.1654575 * np.cos(2.0 * w - 1.261546) - 0.00535383 * np.cos(3.0 * w - 1.1571))
	ts = hh + mn / 60.0 + ss / 3600
	t = 15. * (ts - 12.0 + eta / 60.0) + lon - lon0
	hrangle = t * np.pi / 180
	sinh = np.sin(lat * np.pi / 180) * np.sin(delta) + np.cos(lat * np.pi / 180) * np.cos(delta) * np.cos(hrangle)
	# if sinh < 0:
	# 	hsun = 0
	# else:
	# 	hsun = np.arcsin(sinh)
	hsun = np.arcsin(sinh)
	hsun[sinh<0]=0


	R0 = I00 * d0d2 * np.sin(hsun)

	return R0,hsun


# def MPI_Beta(jday,lat,lon,rh,betaw,betas):
# 	# print(lat,lon)

# 	# !---------------------------------------------------------------------------------------------------
# 	# !
# 	# !     PURPOSE:
# 	# !
# 	# !     Calculate the turbidity coef. according to the dataset of Hess et al. (1998, BAMS)
# 	# !
# 	# !---------------------------------------------------------------------------------------------------
# 	# !
# 	# !     AUTHOR: Kun Yang
# 	# !     24/11/2004 
# 	# !                                                                               
# 	# !     MODIFICATION HISTORY:
# 	# !
# 	# !---------------------------------------------------------------------------------------------------
# 	# print('rh',rh)
# 	rhbeta=np.array((0,50,70,80,90,95,98,99))
# 	# betaw=np.zeros((37,73,8))
# 	# betas=np.zeros((37,73,8))

# 	# data_beta=np.loadtxt('beta_mpi.txt',skiprows=2)

# 	# dims=data_beta.shape
# 	# for i in range(0,8):
# 	# 	for j in range(0,36):
# 	# 		for k in range(0,72):
# 	# 			betaw[j,k,i]=data_beta[j*72+k,i+3]
# 	# 			betas[j,k,i]=data_beta[j*72+k,i+3+8]

# 	# for i in range(0,8):
# 	# 	betaw[:,72,i]=betaw[:,0,i]
# 	# 	betas[:,72,i]=betas[:,0,i]

# 	i = int(lat / 5)
# 	if lat < 0:
# 		i = max(-18, i - 1)
# 	else:
# 		i = min(17, i) 

# 	j = int(lon / 5)
# 	if lon < 0:
# 		j = max(-36, j - 1)
# 	else:
# 		j = min(35, j)


# 	for k in range(1,8):
# 		if(rh >= rhbeta[k-1]) and (rh <= rhbeta[k]):
# 			kk=k
# 	k = min(kk, 7)

# 	################Notice!!!!!!!!!!!!!!
# 	k=k-1
# 	###################


# 	x1 = lat / 5.0 - i
# 	x2 = 1 - x1
# 	y1 = lon / 5.0 - j
# 	y2 = 1 - y1
# 	z1 = (rh - rhbeta[k]) / (rhbeta[k + 1] - rhbeta[k])
# 	z2 = 1 - z1
#     ################### Notice!!!!!!!!!!!!!!
# 	i=i+18
# 	j=j+36
# 	###################


# 	if (jday >= 60) and (jday <= 240):
# 		beta =   betas[i, j, k] * x2 * y2 * z2 + betas[i + 1, j, k] * x1 * y2 * z2 + \
# 		betas[i, j + 1, k] * x2 * y1 * z2 + betas[i, j, k + 1] * x2 * y2 * z1 + \
# 		betas[i + 1, j + 1, k] * x1 * y1 * z2 + betas[i + 1, j, k + 1] * x1 * y2 * z1 + \
# 		betas[i, j + 1, k + 1] * x2 * y1 * z1 + betas[i + 1, j + 1, k + 1] * x1 * y1 * z1
# 	else:
# 		beta =   betaw[i,     j,     k] * x2 * y2 * z2 + betaw[i + 1, j, k] * x1 * y2 * z2 + \
# 		betaw[i, j + 1, k] * x2 * y1 * z2 + betaw[i, j, k + 1] * x2 * y2 * z1 + \
# 		betaw[i + 1, j + 1, k] * x1 * y1 * z2 + betaw[i + 1, j, k + 1] * x1 * y2 * z1 + \
# 		betaw[i, j + 1, k + 1] * x2 * y1 * z1 + betaw[i + 1, j + 1, k + 1] * x1 * y1 * z1

# 	return beta

def MPI_Beta(jday,rh,betaw,betas):
	rhbeta=np.array((0,50,70,80,90,95,98,99.99))
	for i in range(0,len(rhbeta)-1):
		if (rh>=rhbeta[i]) and (rh<rhbeta[i+1]):
			loc=i
	z1=(rh - rhbeta[loc]) / (rhbeta[loc + 1] - rhbeta[loc])
	z2 = 1 - z1

	if (jday >= 60) and (jday <= 240):
		beta=betas[loc]*z2+betas[loc+1]*z1
	else:
		beta=betaw[loc]*z2+betaw[loc+1]*z1

	return beta




def NASA_ozone(jday, lat, lon,year,ozone_nasa):
	# !---------------------------------------------------------------------------------------------------
	# !
	# !     PURPOSE:
	# !
	# !     Calculate the ozone thickness according to the monthly dataset of NASA TOMS.
	# !
	# !---------------------------------------------------------------------------------------------------
	# !
	# !     AUTHOR: Kun Yang
	# !     09/12/2004 
	# !                                                                               
	# !     MODIFICATION HISTORY:
	# !
	# !---------------------------------------------------------------------------------------------------

	# ozone_nasa=np.zeros((37,12))

	# data_ozone=np.loadtxt('ozone_monthly_nasa.txt')
	# lat1=data_ozone[:,0]
	# lat2=data_ozone[:,1]
	# ozone_nasa=data_ozone[:,2:14]
	lat1=np.arange(85,-95,-5)
	lat2=lat1+5
	for i in range(0,len(lat1)):
		if (lat>=lat1[i]) and (lat<=lat2[i]):
			j=i

	t0=dtt.datetime(year,1,1)
	t1=t0+dtt.timedelta(days=jday-1)
	mo=str(t1.strftime('%m'))

	loz = ozone_nasa[j, int(mo)-1]

	return loz







def TRANS(jday,lat,lon,alt,hsun,pa,ta,rh,year,betaw,betas,ozone_nasa):
	# print('calculating the transmittance................')

	# !---------------------------------------------------------------------------------------------------
	# !
	# !     PURPOSE:
	# !
	# !     Calculate the transmittance in clear skies.
	# !
	# !---------------------------------------------------------------------------------------------------
	# !      
	# !     AUTHOR: Kun Yang
	# !     10/11/2004 
	# !                                                                               
	# !     MODIFICATION HISTORY:
	# !
	# !---------------------------------------------------------------------------------------------------


	p0 = 101300

	mass = 1 / (np.sin(hsun) + 0.15 * (57.3 * hsun + 3.885) ** (-1.253))
	# mass = max(0.0, mass)
	mass[mass<0.0]=0.0

	pp0 = pa / p0

	dims=rh.shape
	# print(dims)

	beta=rh*1

	loz=beta*1

	for ii in range(0,dims[0]):
		for jj in range(0,dims[1]):
			
			# beta[ii,jj]=MPI_Beta(jday,lat[ii,jj], lon[ii,jj], rh[ii,jj],betaw,betas)

			betaw_1=betaw[ii,jj,:]
			betas_1=betas[ii,jj,:]
			beta[ii,jj]=MPI_Beta(jday, rh[ii,jj],betaw_1,betas_1)
			loz[ii,jj]=NASA_ozone(jday, lat[ii,jj], lon[ii,jj], year,ozone_nasa)
			# print(beta[ii,jj],loz[ii,jj])

	# beta=MPI_Beta(jday, lat, lon, rh)

	# loz=NASA_ozone(jday, lat, lon, year)

	water = 0.493 * rh / 100.0 / ta * np.exp(26.23 - 5416.0 / ta)

	mp  = mass * pp0
	mb  = mass * beta
	moz = mass * loz
	mw  = mass * water

	lamr = 0.5474 + 0.01424 * mp - 0.0003834 * mp ** 2 + (4.59*10**(-6)) * mp ** 3
	lama = 0.6777 + 0.1464 * mb - 0.00626 * mb ** 2
	koz  = 0.0365 * moz ** (-0.2864)
	gas  = 0.0117 * mp ** 0.3139
	# wv   = -np.log(min(1.0, -0.036 * np.log(mw) + 0.909))
	aaa=-0.036 * np.log(mw) + 0.909
	aaa[aaa>1.0]=1.0
	wv=-np.log(aaa)

	taur  = np.exp(-0.008735 * mp * lamr ** (-4.08))
	taua  = np.exp(-mb * lama ** (-1.3))
	tauoz = np.exp(-moz * koz)
	taug  = np.exp(-gas)
	tauw  = np.exp(-wv) 

	# taub = max (0.0, taur * taua * tauoz * taug * tauw - 0.013)

	taub=taur * taua * tauoz * taug * tauw - 0.013
	taub[taub<0.0]=0.0

	taud = 0.5 * (tauoz * taug * tauw * (1 - taur * taua) + 0.013)

	return taub,taud
	


def Radclr(lon0, lon, lat, alt, yy, mm, dd, hh, mn, ss, pa, ta, rh,betaw,betas,ozone_nasa):
	# print('calculating the radclr................')
	# !---------------------------------------------------------------------------------------------------
	# !
	# !     PURPOSE:
	# !
	# !     This subroutine is used to calculate clear-sky downward solar radiation.
	# !
	# !     Please cite the references:
	# !     Yang, K., Huang, G.-W., & Tamai, N. (2001). A hybrid model for estimating global solar 
	# !           radiation. Solar Energy, 70, 13-22.
	# !     Yang, K., Koike, T., and Ye, B. (2006). Improving estimation of hourly, daily, and monthly 
	# !           solar radiation by importing global data sets. Agricultural and Forest Meteorology, 
	# !           137:43-55.
	# ! 
	# !---------------------------------------------------------------------------------------------------
	# !
	# !     AUTHOR: Kun Yang
	# !     17/08/2005 
	# !                                                                               
	# !     MODIFICATION HISTORY:
	# !
	# !---------------------------------------------------------------------------------------------------
	rad  = 0
	radb = 0
	radd = 0
	Rad0,hsun=solarR0(lon0, lon, lat, yy, mm, dd, hh, mn, ss)

	t_deta=dtt.datetime(yy, mm, dd, hh, mn, ss)-dtt.datetime(yy, 1, 1, 0, 0, 0)
	jday=t_deta.days+1
	taub, taud=TRANS(jday, lat, lon, alt, hsun, pa, ta, rh,yy,betaw,betas,ozone_nasa)
	Radb = Rad0 * taub
	Radd = Rad0 * taud
	Rad  = Radb + Radd
	Rad[Rad0<0.0]=0
	Radb[Rad0<0.0]=0
	Radd[Rad0<0.0]=0

	# if Rad0>0.0:
	# 	t_deta=dtt.datetime(yy, mm, dd, hh, mn, ss)-dtt.datetime(yy, 1, 1, 0, 0, 0)
	# 	jday=t_deta.days+1
	# 	taub, taud=TRANS(jday, lat, lon, alt, hsun, pa, ta, rh,yy)
	# 	Radb = Rad0 * taub
	# 	Radd = Rad0 * taud
	# 	Rad  = Radb + Radd
	# else:
	# 	Rad  = 0 
	# 	Radb = 0
	# 	Radd = 0
	return Rad



def lrd_cal(t_step):

	lons=np.zeros((1339))
	for i in range(0,1339):
		lons[i]=60.86668+0.03333/2+5*0.03333+i*0.03333
	lats=np.zeros((469))
	for i in range(0,469):
		lats[i]=41.46509-0.03333/2-3*0.03333-i*0.03333

	lon,lat=np.meshgrid(lons,lats)

	toph = np.loadtxt('/static_data/ele03333.asc')
	toph=np.delete(toph,[0,1,2],0)
	toph=np.delete(toph,[0,1,2,3,4],1)
	LON_T=0

	######################################
	# betaw=np.zeros((37,73,8))
	# betas=np.zeros((37,73,8))
	# data_beta=np.loadtxt('beta_mpi.txt',skiprows=2)
	# dims=data_beta.shape
	# for i in range(0,8):
	# 	for j in range(0,36):
	# 		for k in range(0,72):
	# 			betaw[j,k,i]=data_beta[j*72+k,i+3]
	# 			betas[j,k,i]=data_beta[j*72+k,i+3+8]

	# for i in range(0,8):
	# 	betaw[:,72,i]=betaw[:,0,i]
	# 	betas[:,72,i]=betas[:,0,i]
	########################################
	kkk=0
	betaw=np.zeros((469,1339,8))
	betas=betaw*1
	for i in (0,50,70,80,90,95,98,99):
		path_betaw='/data/beta_res/betaw_'+'%02d'%i+'.tif'
		beta=read_tif(path_betaw)
		beta=np.delete(beta,[0,1,2],0)
		beta=np.delete(beta,[0,1,2,3,4],1)
		betaw[:,:,kkk]=beta

		path_betas='/data/beta_res/betas_'+'%02d'%i+'.tif'
		beta=read_tif(path_betas)
		beta=np.delete(beta,[0,1,2],0)
		beta=np.delete(beta,[0,1,2,3,4],1)
		betas[:,:,kkk]=beta
		kkk=kkk+1

	########################################
	ozone_nasa=np.zeros((37,12))

	data_ozone=np.loadtxt('ozone_monthly_nasa.txt')
	# lat1=data_ozone[:,0]
	# lat2=data_ozone[:,1]
	ozone_nasa=data_ozone[:,2:14]
	##########################################

	t0=dtt.datetime(1979,1,1,0,0,0)
	t0=t0+dtt.timedelta(days=t_step)
# while t0<=dtt.datetime(2001,1,2,23,0,0):
	yr=str(t0.strftime('%Y'))
	mo=str(t0.strftime('%m'))
	dy=str(t0.strftime('%d'))
	t_str=str(t0.strftime('%Y%m%d'))
	path1='/data/TPMFD/temp/hourly/'+yr+'/tpmfd_temp_h_'+t_str+'_00_23.nc'
	path2='/data/TPMFD/pres/hourly/'+yr+'/tpmfd_pres_h_'+t_str+'_00_23.nc'
	path3='/data/TPMFD/shum/hourly/'+yr+'/tpmfd_shum_h_'+t_str+'_00_23.nc'
	path4='/data/TPMFD/srad/hourly/'+yr+'/tpmfd_srad_h_'+t_str+'_00_23.nc'
	temp=nc_read(path1,'temp')
	pres=nc_read(path2,'pres')
	shum=nc_read(path3,'shum')
	srad=nc_read(path4,'srad')
	ee=shum*pres/(0.622+0.378*shum)
	es=t2es(temp)
	rhum=ee/es*100
	rhum[rhum>99]=99

	crad_d=0
	# tcc=1*rhum
	# tcc[:,:,:]=np.nan
	crad_h=1*rhum
	crad_h[:,:,:]=np.nan
	clf_h1=1*crad_h
	for i in range(0,24):
		print(t_str, i)

		crad=Radclr(LON_T, lon, lat, toph, int(yr), int(mo), int(dy),  i, 0, 0, pres[i,:,:], temp[i,:,:], rhum[i,:,:],betaw,betas,ozone_nasa)
		srad_h=srad[i,:,:]
		srad_h[(srad_h-0.05*crad)<0]=0.05*crad[(srad_h-0.05*crad)<0]
		srad_h[(srad_h-crad)>0]=crad[(srad_h-crad)>0]
		clf_h1[i,:,:]=1-srad_h/crad
		crad_h[i,:,:]=crad

		crad_d=crad_d+crad

	crad_d=crad_d/24
	srad_d=np.mean(srad,0)
	srad_d[(srad_d-0.05*crad_d)<0]=0.05*crad_d[(srad_d-0.05*crad_d)<0]
	srad_d[(srad_d-crad_d)>0]=crad_d[(srad_d-crad_d)>0]



	# tcc_d=np.mean(tcc,0)

	clf_d=1-srad_d/crad_d
	clf_h=1*clf_h1
	for i in range(0,24):
		clf_h0=clf_h1[i,:,:]
		crad_h0=crad_h[i,:,:]
		clf_h0[crad_h0<300]=clf_d[crad_h0<300]
		clf_h[i,:,:]=clf_h0



	Eps=clf_h+(1-clf_h)*1.24*(ee/temp)**(1/7)
	# Eps1=clf_h+(1-clf_h)*(0.5893 + 0.005351 * (ee*100) ** 0.5)
	# Eps[(ee*100)<610]=Eps1[(ee*100)<610]

	lrad_h=(temp**4)*Eps*(5.67*10**(-8))

	nhour=np.zeros((24))
	nhour_c=t0-dtt.datetime(1900,1,1,0,0,0)
	nhour_c=nhour_c.days*24

	for j in range(0,24):
		nhour[j]=nhour_c+j

	path6='/data/TPMFD/lrad_CD99/hourly/'+yr+'/tpmfd_lrad_h_'+t_str+'_00_23.nc'
	print(path6)
	nc_write(path6,lrad_h,nhour)

	# t0=t0+dtt.timedelta(days=1)

if __name__ == '__main__':
	p=Pool(30) 
	for i in range(0,84150): #range(0,1):#
	    r=p.apply_async(lrd_cal,args=(i,)) 
	p.close() #关闭进程池
	p.join()  #结束




	
	





