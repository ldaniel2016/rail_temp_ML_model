### This program tunes the hyperparameters using the Bayes search method

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


def main():

	df =  pd.read_csv(r'data/rail_temperatures_mos_data_sep2019_aug2023.csv', sep = ',')

	

	cols = ['lat', 'lon','analysis_time', 'analysis_date','forecast_time','forecast_period', 
		'T2', 'D2', 'SKT','T_925','WS', 'LCC', 'MCC', 'coshour', 'cosmonth','sinmonth', 
		'sinhour', 'hourly_SRR','hourly_STR', 'TRail']
	df = df[cols]

	df = df.dropna()

	df.TRail = df.TRail + 273.15
	df['month'] = pd.to_datetime(df['forecast_time']).dt.month
	df['year'] = pd.to_datetime(df['forecast_time']).dt.year
	xvar = ['lat', 'lon','forecast_period', 'T2', 'D2', 'SKT','T_925','WS','LCC', 'MCC', 
        'sinhour','coshour', 'sinmonth', 'cosmonth','hourly_SRR','hourly_STR', 'month']

	# Jnauary, April and July 2023 and October 2022 are taken as test data and the remaining is used as train and test in this program.  

	df_test = df[((df['year'] == 2023) & ((df['month']== 1) | (df ['month']== 4) | (df['month']== 7)) | ((df['year'] == 2022) & (df['month']== 10))) ]
	df_train = df[~((df['year'] == 2023) & ((df['month']== 4) | (df ['month']== 5) | (df['month']== 6)| (df['month']== 7)))]


	X = df_train.loc[:, xvar]
	y = df_train.loc[:,'TRail']

	# The training and test data are randomly split on the ratio 0.8:0.2
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


	start_time = datetime.datetime.now()

	optimizer_kwargs = {'acq_func_kwargs':{"xi": 10, "kappa": 10}}
	space  = {'max_depth':Integer(5, 15),
			'learning_rate':Real(0.05, 0.55, "uniform"),
			'colsample_bytree':Real(0.1,1,'uniform'),
			'subsample': Real(0.4, 1, "uniform"),
			'reg_alpha': Real(1e-9, 1,'uniform'),
			'n_estimators': Integer(40, 200)}
	bsearch = BayesSearchCV(estimator = xgb.XGBRegressor(random_state=10), #GradientBoostingRegressor(random_state=10), 
	search_spaces = space, scoring='neg_mean_absolute_error',n_jobs=6, n_iter=100, cv=5, optimizer_kwargs=optimizer_kwargs)
	bsearch.fit(X_test,y_test)
	end_time = datetime.datetime.now()
	elapsed_time = end_time - start_time
	print(elapsed_time)

	dump(bsearch,'Bayes_results_' + 'TRail' + '_new.pkl')
	print("Best Score is: ", bsearch.best_score_, "\n")
	print("Best Parameters: ", bsearch.best_params_, "\n")



if __name__ == "__main__":
	main()


