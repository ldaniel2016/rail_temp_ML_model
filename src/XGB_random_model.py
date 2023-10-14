import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np 

import time
import datetime

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb
from xgboost import XGBRegressor

import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import math
import shap
import joblib


def main():
	df =  pd.read_csv(r'/home/daniel/projects/rails/data/rail_temperatures_mos_data_sep2019_aug2023_corrected_radiation_params.csv', sep = ',')

	cols = ['lat', 'lon','analysis_time', 'analysis_date','forecast_time','forecast_period', 
		'T2', 'D2', 'SKT','T_925','WS', 'LCC', 'MCC', 'coshour', 'cosmonth','sinmonth', 
		'sinhour', 'SRR1h','STR1h', 'TRail']

	df = df[cols]
	df = df.dropna()

	df['month'] = pd.to_datetime(df['forecast_time']).dt.month
	df['year'] = pd.to_datetime(df['forecast_time']).dt.year
	df.TRail = df.TRail + 273.15


	df_test_oct22_jan_aprl_jul23 = df[((df['year'] == 2023) & ((df['month']== 1) | (df ['month']== 4) | (df['month']== 7)) | ((df['year'] == 2022) & (df['month']== 10))) ]
	df_train = df[~((df['year'] == 2023) & ((df['month']== 1) | (df ['month']== 4) | (df['month']== 7)) | ((df['year'] == 2022) & (df['month']== 10)))]
	
	print(df.shape, df_train.shape, df_test_oct22_jan_aprl_jul23.shape)

	xvar = ['lat', 'lon','forecast_period', 'T2', 'D2', 'SKT','T_925','WS','LCC', 'MCC', 
        'sinhour','coshour', 'sinmonth', 'cosmonth','SRR1h','STR1h', 'month']

	# print(df_train.shape, df_test.shape)
	X = df_train.loc[:, xvar]
	y = df_train.loc[:,'TRail']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	print(X_train.shape, X_test.shape)

	# data to plot the RMSE vs forecast period
	df_test = pd.concat([X_test, y_test], axis = 1)

	regressor = xgb.XGBRegressor(n_estimators=156, max_depth=15, eta=0.165, subsample=0.682, colsample_bytree=0.990, reg_alpha=0.140)
	
	# data not normalized
	# sine, cosine hour and month variables 
	# random splitted  test data 
	# all temp are in Kelvin
	# missing values are  removed 

	regressor.fit(X_train, y_train)

	y_trP = regressor.predict(X_train)
	y_tsP = regressor.predict(X_test)

	print('XGB train error:', mean_squared_error(y_train,y_trP,squared=False))
	print('XGB test error:', mean_squared_error(y_test,y_tsP,squared=False))

	# prediction on the unseen test data Oct 2022, Jan, Apr, Jul2023
	X_test_unseen = df_test_oct22_jan_aprl_jul23.loc[:, xvar]
	y_test_unseen = df_test_oct22_jan_aprl_jul23.loc[:, 'TRail']

	y_tsP_unseen = regressor.predict(X_test_unseen)
	print('XGB test error on unseen test data Oct22, Jan, Apr, Jul 2023:', mean_squared_error(y_test_unseen,y_tsP_unseen,squared=False))
	
	df_test_jan23 = df_test_oct22_jan_aprl_jul23[((df_test_oct22_jan_aprl_jul23['year'] == 2023) & (df_test_oct22_jan_aprl_jul23['month']== 1))]
	X_jan23 = df_test_jan23.loc[:, xvar]
	y_jan23 = df_test_jan23.loc[:, 'TRail']

	y_tsP_jan23 = regressor.predict(X_jan23)
	print('XGB test error on unseen Jan23:', mean_squared_error(y_jan23,y_tsP_jan23,squared=False))

	df_test_apr23 = df_test_oct22_jan_aprl_jul23[((df_test_oct22_jan_aprl_jul23['year'] == 2023) & (df_test_oct22_jan_aprl_jul23['month']== 4))]
	X_apr23 = df_test_apr23.loc[:, xvar]
	y_apr23 = df_test_apr23.loc[:, 'TRail']
	y_tsP_apr23 = regressor.predict(X_apr23)
	print('XGB test error on unseen Apr23:', mean_squared_error(y_apr23,y_tsP_apr23,squared=False))

	df_test_jul23 = df_test_oct22_jan_aprl_jul23[((df_test_oct22_jan_aprl_jul23['year'] == 2023) & (df_test_oct22_jan_aprl_jul23['month']== 7))]
	X_jul23 = df_test_jul23.loc[:, xvar]
	y_jul23 = df_test_jul23.loc[:, 'TRail']
	y_tsP_jul23 = regressor.predict(X_jul23)
	print('XGB test error on unseen Jul23:', mean_squared_error(y_jul23,y_tsP_jul23,squared=False))

	df_test_oct22 = df_test_oct22_jan_aprl_jul23[((df_test_oct22_jan_aprl_jul23['year'] == 2022) & (df_test_oct22_jan_aprl_jul23['month']== 10))]
	X_oct22 = df_test_oct22.loc[:, xvar]
	y_oct22 = df_test_oct22.loc[:, 'TRail']
	y_tsP_oct22 = regressor.predict(X_oct22)
	print('XGB test error on unseen Oct22:', mean_squared_error(y_oct22,y_tsP_oct22,squared=False))


	# save the model
	param = 'TRail'
	joblib.dump(regressor, '/home/daniel/projects/rails/results/xgb_random_model_corrected_radiation_params_' + param + '.joblib')

	# to plot the RMSE vs lead time for not normalized data

	ajat = sorted(df_test['forecast_period'].unique().tolist())


	xgb_error = [None] * len(ajat)
	for i in range(0,len(ajat)):
		tmp = df_test[df_test['forecast_period'] == ajat[i]]
		df_x_test_ajat = tmp.drop('TRail', axis =1)
		x_test_ajat = df_x_test_ajat
		y_test_ajat = tmp['TRail']
		y_xgb_ajat = regressor.predict(x_test_ajat)
		xgb_error[i] = mean_squared_error(y_test_ajat,y_xgb_ajat,squared=False)

	plt.close()
	plt.plot(ajat, xgb_error, 'b', label="XGB")
	plt.title("XGB RMSE")
	plt.xlabel("forecast_period")
	plt.ylabel("RMSE")
	plt.legend(loc="lower right")

	
	plt.savefig('XGB_forecast_period_corrected_radiation_params_' + param + '_random_split_testrmse_basic_14Oct2023.png')

	#Shap values for nnot ormalized data
	import shap
	# take random set from test data
	X_sub = X_test.sample(frac=0.1)

	shap_values = shap.Explainer(regressor).shap_values(X_sub)
	plt.close()
	shap.summary_plot(shap_values, X_sub, plot_type="bar", show = False)

	plt.savefig('shap_bar_corrected_radiation_params_' + 'TRail' + 'random_split_basic_14Oct2023' + '.png')

if __name__ == "__main__":
	main()


# Results
# XGB train error: 1.7342630690383463
# XGB test error: 2.2712756447146014
# XGB test error on unseen test data Oct22, Jan, Apr, Jul 2023: 4.368814056717018
# XGB test error on unseen Jan23: 2.5716137031372726
# XGB test error on unseen Apr23: 6.36059249687501
# XGB test error on unseen Jul23: 3.4737688736892025
# XGB test error on unseen Oct22: 2.791425315238639



