import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np 

import time
import datetime

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
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
import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import math
import shap

import skopt
from skopt.space import Real, Integer, Categorical
from skopt import dump, load
import sys,os,getopt
import matplotlib.pyplot as plt
import seaborn as sns
import math
import xgboost as xgb
from xgboost import XGBRegressor
#from MLmodify import modify 
from skopt import BayesSearchCV 
import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')


def on_step(optim_result):
	#Callback meant to view scores after each iteration while performing Bayesian
	#Optimization in Skopt
	score = opt.best_score_
	print("best score: %s" % score)
	if score >= -0.11:
		print('Interrupting!')
		return True

def main():

	df =  pd.read_csv(r'/home/daniel/projects/rails/data/rail_temperatures_mos_data_sep2019_aug2023_corrected_radiation_params.csv', sep = ',')

	

	cols = ['lat', 'lon','analysis_time', 'analysis_date','forecast_time','forecast_period', 
		'T2', 'D2', 'SKT','T_925','WS', 'LCC', 'MCC', 'coshour', 'cosmonth','sinmonth', 
		'sinhour', 'SRR1h','STR1h', 'TRail']
	df = df[cols]

	df = df.dropna()

	df.TRail = df.TRail + 273.15
	df['month'] = pd.to_datetime(df['forecast_time']).dt.month
	df['year'] = pd.to_datetime(df['forecast_time']).dt.year

	xvar = ['lat', 'lon','forecast_period', 'T2', 'D2', 'SKT','T_925','WS','LCC', 'MCC', 
        'sinhour','coshour', 'sinmonth', 'cosmonth','SRR1h','STR1h', 'month']

	df_test_oct22_jan_aprl_jul23 = df[((df['year'] == 2023) & ((df['month']== 1) | (df ['month']== 4) | (df['month']== 7)) | ((df['year'] == 2022) & (df['month']== 10))) ]
	df_train = df[~((df['year'] == 2023) & ((df['month']== 1) | (df ['month']== 4) | (df['month']== 7)) | ((df['year'] == 2022) & (df['month']== 10)))]

	print(df.shape, df_train.shape, df_test_oct22_jan_aprl_jul23.shape)
	X = df_train.loc[:, xvar]
	y = df_train.loc[:,'TRail']

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	
	
	


	start_time = datetime.datetime.now()
	print(start_time)

	optimizer_kwargs = {'acq_func_kwargs':{"xi": 10, "kappa": 10}}
	space  = {'max_depth':Integer(5, 15),
			'learning_rate':Real(0.05, 0.55, "uniform"),
			'colsample_bytree':Real(0.1,1,'uniform'),
			'subsample': Real(0.4, 1, "uniform"),
			'reg_alpha': Real(1e-9, 1,'uniform'),
			'n_estimators': Integer(40, 160)}
	bsearch = BayesSearchCV(estimator = xgb.XGBRegressor(random_state=10), #GradientBoostingRegressor(random_state=10), 
	search_spaces = space, scoring='neg_mean_absolute_error',n_jobs=6, n_iter=100, cv=5, optimizer_kwargs=optimizer_kwargs)
	bsearch.fit(X_test,y_test)
	#parameter_over_iterations(bsearch)
	end_time = datetime.datetime.now()
	elapsed_time = end_time - start_time
	print(elapsed_time)

	dump(bsearch,'/home/daniel/projects/rails/results/results_corrected_radiation' + 'TRail' + '_new.pkl')
	print("Best Score is: ", bsearch.best_score_, "\n")
	print("Best Parameters: ", bsearch.best_params_, "\n")


"""

	regressor = xgb.XGBRegressor(n_estimators=100, max_depth=10, eta=0.1, subsample=0.7, colsample_bytree=0.8)

	xgbr = xgb.XGBRegressor(objective = "reg:squarederror", n_jobs = -1, random_state = 0)
	start = datetime.datetime.now()
	print(start)
	kf = KFold(n_splits = 4, shuffle = True, random_state = 0)
	opt = BayesSearchCV(xgbr, 
			{
			"learning_rate": Real(0.01, 1.0, 'uniform'),
			"n_estimators": Integer(50, 100),
			"max_depth": Integer(3, 13),
			"colsample_bytree": Real(0.1, 1,'uniform'),
			"subsample": Real(0.1, 1,'uniform'),
			"reg_alpha": Real(1e-9, 1,'uniform'),
			"reg_lambda": Real(1e-9, 1,'uniform'),
			"gamma": Real(0, 0.5)
			},
		n_iter = 10,  
		cv = kf,
		n_jobs = -1,
		scoring = "neg_root_mean_squared_error",
		random_state = 0
		)
	res = opt.fit(X_test, y_test)
	dump(res,'results_wg.pkl')      

	end = datetime.datetime.now()
	elapsed_time = end - start
	print(elapsed_time)
	print("Best Score is: ", opt.best_score_, "\n")
	print("Best Parameters: ", opt.best_params_, "\n")

	dump(res,'/home/daniel/projects/rails/data/optimized_results_' + 'TRail'+ '_new.pkl')
"""

if __name__ == "__main__":
	main()


