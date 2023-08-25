import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def surf_ref(ts,sr,pgt, peak_pow, gr):
	'''to calculate the surface reflectivity
	ts - range between the transmitter antenna and the surface
	sr - range between the surface and the receiver antenna
	pgt - product of power of the transmitter antenna and transmitter antenna gain 
	peak_pow - peak power of the delay doppler map
	gr - receiver antenna gain'''
	t_dist = [sum(value) for value in zip(ts, sr)] # total distance
	num_list1 = np.multiply(t_dist,t_dist) # square of total distance
	# multiplying according to the bistatic radar equation
	mul_val = [[[1/n for n in num] for num in sublist] for sublist in np.multiply(pgt,gr)]
	SR1 = np.multiply(np.multiply(np.multiply(num_list1,16*np.pi**2/0.19**2),peak_pow), mul_val)
	SR = [[[10*np.log10(n) for n in num] for num in sublist] for sublist in SR1]
	return SR

def lon_lon_conv(lon):
	# to convert from 0 to 360 longitdue to -180 to 180 degree longitude
	return np.array(lon, dtype = object) - 360

def grid_coord(latitude, longitude):
	act_lat = np.arange(latitude[0] + abs((latitude[1]-latitude[0])/2), latitude[-1] - abs((latitude[1]-latitude[0])/2), (latitude[1]-latitude[0]))

	act_lon = np.arange(longitude[0] - abs((longitude[1]-longitude[0])/2), longitude[-1] + abs((longitude[1]-longitude[0])/2), (longitude[1]-longitude[0]))

	return act_lat, act_lon

def data_pd(lat, lon, SR, r_ant_gain, inc_ang, SNR, n_files, all_days): # data in pandas dataframe
	'''lat - latitude of the data points
	lon - longitdue of the data points
	SR - surface reflectivity
	r_ant_gain - recevier antenna gain
	inc_ang - incident angle of signal from the surface
	SNR - Signal to noise ratio
	n_files - number of files to be read
	all_days - the days of the respective files'''
	longitude = []
	latitude = []
	su_ref = []
	ant_gain = []
	inc = []
	sig_noise = []
	for i in range(n_files):
	    latitude.append(np.reshape(lat[i],(-1))) # latitude
	    longitude.append(np.reshape(lon[i],(-1))) # longitude
	    su_ref.append(np.reshape(SR[i],(-1))) # surface reflectivity
	    ant_gain.append(np.reshape(r_ant_gain[i],(-1))) # receiver antenna gain 
	    inc.append(np.reshape(inc_ang[i],(-1))) # incident angle of signal from the surface
	    sig_noise.append(np.reshape(SNR[i],(-1))) # signal to noise ratio
	     
	data = pd.DataFrame({"lat":latitude, "lon":longitude, "SR":su_ref, "inc_ang":inc, "SNR":sig_noise, "gr":ant_gain, "Date": all_days})

	return data

def dt_otl_grid(data, nfiles, xx_grid, yy_grid, SNR = False):
	'''To compute the calibrated datasets in the grid'''
	sr = []
	days = []
	date=[]
	snr = []

	for k in range(nfiles):
	    ad = pd.DataFrame({"SR":data["SR"][k],"lat":data["lat"][k],"lon":data["lon"][k], "inc_ang":data["inc_ang"][k], "gr":data["gr"][k], "SNR":data["SNR"][k], "Date":data["Date"][k]})
	    ad = ad[~np.isnan(ad["SR"])]
	    
	    # the calibration is as follows
	    filtered_data1 = ad[ad["SNR"] >= 2]
	    filtered_data1 = filtered_data1[filtered_data1["gr"] >= 0]
	    filtered_data1 = filtered_data1[filtered_data1["inc_ang"] <= 65]
	    
	    df = pd.DataFrame({"lat": filtered_data1["lat"], "lon": filtered_data1["lon"]})
	        
	    dat_grid_SR = griddata(df, filtered_data1["SR"], (yy_grid, xx_grid), method='nearest')
	    
	    sr.append(dat_grid_SR.reshape(-1))
	    
	    date.append(filtered_data1["Date"])

	    if SNR == True:
	    
		    dat_grid_SNR = griddata(df, filtered_data1["SNR"], (yy_grid, xx_grid), method='nearest')
		    
		    snr.append(dat_grid_SNR.reshape(-1))

	return sr, date, snr

def st_dev_change(data, nfiles, plt_req = True, all_plt = False):
	'''To compute the standard deviation of the values before and after the removal of some empirical calibrations
	data - data in pandas data frame
	nfiles - number of files
	plt_req - mean of all the standard deviations values
	all_plt - standard deviation of all the points'''

	std_dev_bef =[]
	std_dev_aft = []
	mean_val_aft = []
	mean_val_bef = []

	for k in range(nfiles):
	    ad = pd.DataFrame({"SR":data["SR"][k],"lat":data["lat"][k],"lon":data["lon"][k], "inc_ang":data["inc_ang"][k], "gr":data["gr"][k], "SNR":data["SNR"][k], "Date":data["Date"][k]})
	    ad = ad[~np.isnan(ad["SR"])]
	    
	    std_bef = np.std(ad['SR']) # standard deviation before calibrating

	    filtered_data1 = ad[ad["SNR"] >= 2] # to remove the values which have SNR < 2

	    filtered_data1 = filtered_data1[filtered_data1["gr"] >= 0] # to remove the valuyes which have receiver antenna less than 0

	    filtered_data1 = filtered_data1[filtered_data1["inc_ang"] <= 65] # to remove the values whose incident angles are more than 65 degrees
	    
	    std_aft = np.std(filtered_data1['SR']) # standard deviation after calibrating

	    std_dev_bef.append(std_bef)
	    std_dev_aft.append(std_aft)

	mean_std_dev_aft = np.mean(std_dev_aft) # mean of all the standard deviation after calibrating
	mean_std_dev_bef = np.mean(std_dev_bef) # mean of all the standard deviation before calibrating

	if plt_req == True:
		plt.plot(["Before Outlier Removal", "After Outlier Removal"], [mean_std_dev_bef,mean_std_dev_aft], marker = ".")
		plt.title("Average Standard Deviation")
		plt.show()
	if all_plt == True:
		plt.plot(["Before Outlier Removal", "After Outlier Removal"], [std_dev_bef,std_dev_aft], marker = ".")
		plt.title("Standard Deviation of each Satellite Observation for all days")
		plt.show()

	return (mean_std_dev_aft-mean_std_dev_bef)*100/mean_std_dev_bef

def data_date(date): # to get one date value for a day
	date1 = []
	for i in date:
	    if len(i)==0:
	        continue
	    else:
	        dates1 = list(i)[0]
	    date1.append(dates1) # to get the date values
	return date1

def linear_reg(X,Y, P = 1): 
    '''to perform the linear regression
    X - the X coordinate to be changed
    Y - the Y coordiante to be changed'''
    des_mat = []
    [des_mat.append([i,1]) for i in X] # design matrix

    N = np.dot(np.dot(np.array(des_mat).T, P), des_mat) # normal matrix

    U = np.dot(np.array(des_mat).T,Y) # right hand vector

    unknown_X = np.dot(np.linalg.inv(N),U) # unknown paramters

    V = np.dot(des_mat,unknown_X) - Y # residual matrix

    adj_Y = Y + V # adjusted values

    return adj_Y, unknown_X[0], unknown_X[1] # adjusted value, slope and constant values

def min_max(data):
	'''to normalise the data for getting values in same scale
	data - the data to be nromalised'''
	if np.isnan(data).any(): # to check if there is any np.nan value in the dataset
		first11 = np.nanmax(data)
		first_1 = np.nanmin(data)
		reflect111 = (data - first_1)/(first11-first_1)
	else:
		first11 = np.max(data)
		first_1 = np.min(data)
		reflect111 = (data - first_1)/(first11-first_1)
	return reflect111

def corr_coeff(grids, f_data, s_data):
	'''to compute the correlation coefffcient
	grids - total grids in the gridded region
	f_data - soil moisture
	s_data - surface reflectivity'''
	coeff = []

	for i in range(grids):
	    
	    ak = pd.DataFrame({"SM":f_data[i], "SR":s_data[i]}) # to form a pandas dataframe
	    
	    ak1 = ak[~np.isnan(ak["SR"].astype(float))] # to remove the np.nan values
	    ak2 = ak1[~np.isnan(ak1["SM"].astype(float))]
	    
	    ak3 = ak2.reset_index(drop=True) 

	    reflect111 = min_max(ak3['SR']) # normalising
	    SM111 = min_max(ak3["SM"])
	    
	    # formula for computing the correlation coeffcient
	    hh = len(SM111) * sum(SM111**2) - sum(SM111)**2
	    gg = len(reflect111) * sum(reflect111**2) - sum(reflect111)**2

	    jj = len(reflect111) * sum(reflect111*SM111) - sum(reflect111) * sum(SM111)
	    
	    coeff.append(jj/np.sqrt(hh*gg))

	return coeff

def time_agg(days, data, lon_grid, lat_grid, cmap, typ = "n_shp", plot = True, lat_shape = 0, lon_shape = 0, Title = "", s_data = None):
	'''averaging over days and plotting
	days - the range of days
	data - the data that is to be averaged
	lon_grid - X coordinates from meshgrid
	lat_grid - Y coordinates from meshgrid
	cmap - colormap
	typ - is the data with respect to the shape file or not
	plot - if plot is required or not
	lat_shape - required latitude shape
	lon_shape - required longitdue shape
	title - title to be given
	s_data - used if the required data is not with respect to the shape file but the s_data is'''
	ab1 = []
	j = 0
	for i in range(len(days)):

		if typ == "shp": # if data is with respect to shape file
			abcd2 = np.nanmean(data[j:j+days[i]],axis=0)
		elif typ == "n_shp": # if data is not with respect to shape file
			fi = np.mean(data[j:j+days[i]],axis=0)
		    
			si = np.nanmean(s_data[j:j+days[i]],axis=0)

			abcd2 = fi+si-si # to crop the data according to the shaep file

		ab_b = np.nanmean(abcd2) # to compute the mean value of entire
		    
		ab1.append(ab_b)

		SM111 = min_max(abcd2) # normalising
		    
		if plot == True:

			ax = plt.contourf(lon_grid,lat_grid,np.reshape(SM111,(lat_shape,lon_shape)),cmap = cmap) 
			plt.title(f'{Title}', fontsize=14,pad = 10)  
			plt.contourf(lon_grid,lat_grid,np.reshape(SM111,(lat_shape,lon_shape)),cmap = cmap) # plotting the sea surface temperature of each month
			plt.colorbar(extend='both')
			plt.show()
		    
		j = j+days[i]
	return ab1