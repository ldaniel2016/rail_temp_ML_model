# This program develops the ML model for predictiong the rail temperature

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import time

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
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the rail temperature data for the Väylävirasto stations along with 
# the forecast data

df =  pd.read_csv(r'/home/daniel/projects/rails/data/rail_temperatures_mos_data.csv', sep = ',')

# changing the float64 data types to float32 data
cols = df.select_dtypes(include=[np.float64]).columns
df[cols] = df[cols].astype(np.float32)


# changing the int64 data types to int32 data
cols = df.select_dtypes(include=[np.int64]).columns
df[cols] = df[cols].astype(np.int32)


# converting T2, D2 and SKT from degree K to degree celsius
df.T2= df['T2'] - 273.15
df.D2 = df['D2'] - 273.15
df.SKT = df['SKT'] - 273.15

# The predictors for the model
xvar = ['lat', 'lon','forecast_period', 'T2', 'D2', 'SKT','U10', 'V10', 
        'sinhour', 'hourly_SRR','hourly_STR', 'hourly_SLHF', 
        'hourly_SSHF']


X = df.loc[:, xvar]
y = df.loc[:,'TRail']

# Splitting the data to training and test samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# This dataframe we use to plot the graph of RMSE vs forecast_period
df_test = pd.concat([X_test, y_test], axis = 1)

# create scaling object based on X_train and scale both X_train and X_text
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#fit scaler to training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Training the XGB model
model  = xgb.XGBRegressor(n_estimators=100, max_depth=11, eta=0.3, subsample=0.7, colsample_bytree=0.8)

# Predicting using the xgb_model
model.fit(X_train, y_train)

y_trP = model.predict(X_train)
y_tsP = model.predict(X_test)

print('XGB train error:', mean_squared_error(y_train,y_trP,squared=False))
print('XGB test error:', mean_squared_error(y_test,y_tsP,squared=False))


# To plot the RMSE vs lead time
ajat = sorted(df_test['forecast_period'].unique().tolist())
xgb_error = [None] * len(ajat)

for i in range(0, len(ajat)):
	tmp = df_test[df_test['forecast_period'] == ajat[i]]
	df_x_test_ajat = tmp.drop('TRail', axis =1)

	x_test_ajat = scaler.transform(df_x_test_ajat)
	y_test_ajat = tmp['TRail']
	y_xgb_ajat  = model.predict(x_test_ajat)
	xgb_error[i] = mean_squared_error(y_test_ajat, y_xgb_ajat, squared=False)

plt.plot(ajat, xgb_error, 'b', label="XGB")
plt.title("XGB RMSE")
plt.xlabel("forecast_period")
plt.ylabel("RMSE")
plt.legend(loc="lower right")

param = 'TRail'
plt.savefig('XGB_forecast_period_' + param + '_rmse.png')

# save the model using joblib
joblib.dump(model, "xgb_modelv1.joblib")
print('The xgbmodel is saved as xgb_modelv1.joblib')
