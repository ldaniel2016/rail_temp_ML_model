### This program trains the ML model for predicting the rail_track temperature

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
	df =  pd.read_csv(r'data/rail_temperatures_mos_data_sep2019_aug2023.csv', sep = ',')

	cols = ['lat', 'lon','analysis_time', 'analysis_date','forecast_time','forecast_period', 
		'T2', 'D2', 'SKT','T_925','WS', 'LCC', 'MCC', 'coshour', 'cosmonth','sinmonth', 
		'sinhour', 'hourly_SRR','hourly_STR', 'TRail']
	df = df[cols]
	df = df.dropna()

	df['month'] = pd.to_datetime(df['forecast_time']).dt.month
	df['year'] = pd.to_datetime(df['forecast_time']).dt.year
	df.TRail = df.TRail + 273.15


	# data not normalized
    # sine, cosine hour and month variables 
    # random splitted training and  test data 
    # all temperature variables are in  Kelvin
    # missing values are  removed 

	# this df_test dataframe is created for plotting 
	df_test = pd.concat([X_test, y_test], axis = 1)

	regressor = xgb.XGBRegressor(n_estimators=200, max_depth=15, eta=0.182, subsample=0.751, colsample_bytree=0.997, reg_alpha=0.54)
	
	regressor.fit(X_train, y_train)

	y_trP = regressor.predict(X_train)
	y_tsP = regressor.predict(X_test)

	print('XGB train error:', mean_squared_error(y_train,y_trP,squared=False))
	print('XGB test error:', mean_squared_error(y_test,y_tsP,squared=False))


	# save the model
	param = 'TRail'
	joblib.dump(regressor, '../results/xgb_random_model' + param + '.joblib')

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

    param = 'TRail'	
	plt.savefig('XGB_forecast_period_' + param + '.png')

	#Shap values for nnot ormalized data
	import shap
	# take random set from test data
	X_sub = X_test.sample(frac=0.1)
	print(X_sub.size)
	shap_values = shap.Explainer(regressor).shap_values(X_sub)
	plt.close()
	shap.summary_plot(shap_values, X_sub, plot_type="bar", show = False)

	plt.savefig('shap_bar_' + param '.png')


if __name__ == "__main__":
	main()


# Results
# XGB train error: 1.6340721353575478
# XGB test error: 1.6393303799561285
# XGB test error on so far unforeseen data, i.e, Oct 2022 and Jan,Apr, Jul 2023
# XGB testing error: 4.1234929752605565
# Jan 2023
# XGB testing error: 1.1988764521942834
# April 2023
# XGB testing error: 6.566499205956574
# July 2023
# XGB testing error: 3.5533725085060115
# October 2022
# XGB testing error: 1.6019979771691122

