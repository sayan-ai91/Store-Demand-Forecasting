# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:05:51 2020

@author: Sayan Mondal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, time
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

data=pd.read_csv("C:/Users/Sayan Mondal/Desktop/demand-forecasting-kernels-only/train.csv")

data['date'].dtype
data.head()
data.info()

data.describe()
data.isna().sum()

data['sales'].min()
data['sales'].max()

data.hist()

plt.figure(figsize=(16,5))
plt.title("Distribution of sales")
ax = sns.distplot(data['sales'])

## sales distribution itemwise...##
rcParams['figure.figsize'] = 16, 7
sns.barplot(x=data['item'], y=data['sales'], errwidth=0,palette="PuBuGn_d")
plt.title('Sales distribution across items')

## sales distribution storewise...##
sns.barplot(x=data['store'], y=data['sales'], errwidth=0,palette="BuGn_r")
plt.title('Storewise sales distribution')


## QQ PLOT...##
import scipy.stats
import pylab

scipy.stats.probplot(data.sales,plot=pylab)
pylab.show()

###converting string input to datetime(int) value..##
data.date=pd.to_datetime(data.date,dayfirst=True)
data.head()
data.date.describe()
data.date.dtype

## extraction of year,month,week & day from the Date separately...###
data['year'] = data.date.dt.year
data['month']=data.date.dt.month
data['week']=data.date.dt.week
data['day']=data.date.dt.day

## sales distribution yearly...##
sns.barplot(x=data['year'], y=data['sales'], errwidth=0,palette="GnBu_d")
plt.title('Yearly sales distribution')

## sales distribution monthly...##
sns.barplot(x=data['month'], y=data['sales'], errwidth=0,palette="Blues")
plt.title('Monthly sales distribution')

## sales distribution weekly...##

sns.barplot(x=data['week'], y=data['sales'], errwidth=0,palette="husl")
plt.title('weekly sales distribution')

## Checking month & year wise Sales plot..
sns.lineplot(x=data['month'],y=data['sales'],data=data)

sns.lineplot(x=data['year'],y=data['sales'],data=data)

## Now lets put date as index value, by doing that date will no longer be a separate attribute..##
data.set_index("date",inplace=True)
data.head()

############# seasonal decomposition of dataset...########################
from pylab import rcParams

rcParams['figure.figsize'] = 16, 8
figure = sm.tsa.seasonal_decompose(data.sales[:730],freq=365).plot() # decompose with 365
figure.show()

##so from the above fig. it clearly shows data has seasonality...

## Rolling Stat...### moving average...##
data_ma= data.sales.rolling(window=15).mean()
data_ma[:730].plot()

##...ACF PLOT For original sales data..##
rcParams['figure.figsize'] = 10,5
plot_acf(data.sales[:1000],lags=20)


## Lets convert the series to a stationary one..####
sales_diff_1= data.sales.diff(periods=1)## integrated order of 1, denoted by 'd'...## parameters of ARIMA Model..###
#sales_diff_1.dropna(inplace=True) ### removing the Nan value from the series..##
sales_diff_1.head()
#sales_diff_1[:100].plot()

##Creating new column for sales diiff of 1 lag..##
data['shift1']=sales_diff_1

## sales difference with 2 lag...##
sales_diff_2= data.sales.diff(periods=2)
sales_diff_2[:365].plot()

##### ACF & PACF plot...###

rcParams['figure.figsize'] = 14, 5
figure = plot_acf(data.shift1[1:10000],lags=30,title='Sales ACF plot- 30lags') ## 
figure.show()

rcParams['figure.figsize'] = 14, 5
figure = plot_pacf(data.shift1[1:10000],lags=25,title='Sales PACF plot- 25lags') ## 
figure.show()




####........MODEL building.## AR model...#######################

data=data.dropna(axis=0, inplace=False)

X=data.sales.values ## .values returns list of all 
X.shape
train=X[0:3000]
train.size
test=X[3000:3365]
test.size


from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error as mse


model_ar=AR(train)
model_ar_fit=model_ar.fit()
 
predict=model_ar_fit.predict(start=3000,end=3364)
predict.size

##RMSE....####
mse(test,predict)
np.sqrt(236.77386529873033)##15.38 RMSE

###SYMMETRIC MAPE...###
EPSILON = 1e-10
def SMAPE(test, predict):
    """Symmetric Mean Absolute Percentage Error"""
    return np.mean((2.0 * np.abs(test - predict) / ((np.abs(test) + np.abs(predict)) + EPSILON)))*100
    
SMAPE(test,predict) ### 26.252...

plt.plot(test[:100])
plt.plot(predict[:100],color='green')
plt.show()

### ARIMA MODEL....######
from statsmodels.tsa.arima_model import ARIMA

model_arima=ARIMA(train,order=(6,1,5))

model_arima_fit=model_arima.fit()

model_arima_fit.summary()

predict=model_arima_fit.forecast(steps=365)[0]

##RMSE....####
mse(test,predict)
np.sqrt(71.61283058908056) ## 8.46...

###SYMMETRIC MAPE...###
EPSILON = 1e-10
def SMAPE(test, predict):
    """Symmetric Mean Absolute Percentage Error"""
    return np.mean((2.0 * np.abs(test - predict) / ((np.abs(test) + np.abs(predict)) + EPSILON)))*100
    
SMAPE(test,predict) ### 22.94..

plt.plot(test,color='orange')
plt.plot(predict,color='black')
plt.show()

import itertools
p=d=q=range(1,9)
pdq=list(itertools.product(p,d,q))

import warnings
warnings.filterwarnings('ignore')
for param in pdq:
    try:
        model_arima=ARIMA(train,order=param)
        model_arima_fit=model_arima.fit()
        print(param,model_arima_fit.aic)
    except:
        continue
    

##############............SARIMA Model......#############################
        
from statsmodels.tsa.statespace.sarimax import SARIMAX

model_sarima=SARIMAX(train,order=(6,1,5),seasonal_order=(1,1,1,7))

model_sarima_fit=model_sarima.fit(disp=False)

model_sarima_fit.summary()

predict=model_sarima_fit.predict(start=3000,end=3364)

##RMSE....####
np.sqrt(mse(test,predict))...##8.43


###SYMMETRIC MAPE...###
EPSILON = 1e-10
def SMAPE(test, predict):
    """Symmetric Mean Absolute Percentage Error"""
    return np.mean((2.0 * np.abs(test - predict) / ((np.abs(test) + np.abs(predict)) + EPSILON)))*100
    
SMAPE(test,predict) ### 22.63..

plt.plot(test,color='green')
plt.plot(predict,color='black')
plt.show()

################.......... AUTO ARIMA Model.........###########################
from pmdarima.arima import auto_arima

stepwise_model = auto_arima(train, start_p=1, start_q=1,
                           max_p=8, max_q=9, m=7,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
""" auto arima dose't need separately fit function it directly fits the data"""

print(stepwise_model.aic())  .....##AIC 18197.167468051142...#

stepwise_model.summary()

predict= stepwise_model.predict(n_periods=365)


### RMSE value.....####
mse(test,predict)
np.sqrt(70.6234343110864)..## 8.40

###SYMMETRIC MAPE...###
EPSILON = 1e-10
def SMAPE(test, predict):
    """Symmetric Mean Absolute Percentage Error"""
    return np.mean((2.0 * np.abs(test - predict) / ((np.abs(test) + np.abs(predict)) + EPSILON)))*100
    
SMAPE(test,predict) ### 22.33..

rcParams['figure.figsize'] = 12, 5
plt.plot(test,color='#15e61f')
plt.plot(predict,color='#f70a16')
plt.show()



################## RNN-LSTM model.....###################################

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional

scaler=MinMaxScaler(feature_range=(0,1))

X=scaler.fit_transform(np.array(X).reshape(-1,1))

X.shape
print(X)

train=X[0:3000]
train.size
test=X[3000:3365]
test.size

#####.....Creating a dataset with 25 timesteps and 1 output...#########

"""X_train=[]
y_train=[]
for i in range(25,3000):
    X_train.append(X[i-25:i,0])
    y_train.append(X[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)

print(X_train.shape)
print(y_train.shape)    """

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


time_step=100
X_train,y_train=create_dataset(train,time_step)
X_test,y_test=create_dataset(test,time_step)

print(X_train.shape)
print(y_train.shape) 
print(X_test.shape)
print(y_test.shape)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


## Model Building..###
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(Dropout(0.2))
model.add(LSTM(50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=30,batch_size=100,verbose=1)

### Lets check performence of model...##
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transform back to its original form....##
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


## checking the RMSE value for both Train & Test...###

mse(y_train,train_predict)
np.sqrt(mse(y_train,train_predict))...##22.83

np.sqrt(mse(y_test,test_predict))#28.59


###SYMMETRIC MAPE...###
EPSILON = 1e-10
def SMAPE(y_test, test_predict):
    """Symmetric Mean Absolute Percentage Error"""
    return np.mean((2.0 * np.abs(y_test - test_predict) / ((np.abs(y_test) + np.abs(test_predict)) + EPSILON)))*100
    
SMAPE(y_test,test_predict)## 198.17



######################### CAT BOOST Algorithm...################################

X=data.sales.values ## .values returns list of all

scaler=MinMaxScaler(feature_range=(0,1))

X=scaler.fit_transform(np.array(X).reshape(-1,1))
train=X[0:20000]
train.size
test=X[20000:21000]
test.size

##### convert an array of values into a dataset matrix..##########
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


time_step=100
X_train,y_train=create_dataset(train,time_step)
X_test,y_test=create_dataset(test,time_step)

print(X_train.shape)
print(y_train.shape) 
print(X_test.shape)
print(y_test.shape)

from catboost import CatBoostRegressor

cb=CatBoostRegressor()


model=cb.fit(X_train,y_train)

y_pred=model.predict(X_test)

np.sqrt(mse(y_test,y_pred))###..0.0444..##


EPSILON = 1e-10
def SMAPE(y_test, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return np.mean((2.0 * np.abs(y_test - y_pred) / ((np.abs(y_test) + np.abs(y_pred)) + EPSILON)))*100
    
SMAPE(y_test, y_pred) ##11.854..###



################ Test data...### Finalized model is CatBoost...#####################

data1=pd.read_csv("C:/Users/Sayan Mondal/Desktop/demand-forecasting-kernels-only/test.csv")

###converting string input to datetime(int) value..##
data1.date=pd.to_datetime(data1.date,dayfirst=True)
#data1.set_index("date",inplace=True)

## Joining two dataset(train & test) togeather...###

frames=[data,data1]

dataset=pd.concat(frames)

X=dataset.sales.values ## .values returns list of all...958000 total data..

scaler=MinMaxScaler(feature_range=(0,1))

X=scaler.fit_transform(np.array(X).reshape(-1,1))
train=X[0:30000]
train.size
test=X[912899:958000]
test.size

##### convert an array of values into a dataset matrix..##########
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


time_step=100
X_train,y_train=create_dataset(train,time_step)
X_test,y_test=create_dataset(test,time_step)


print(X_train.shape)
print(y_train.shape) 
print(X_test.shape)
print(y_test.shape)

from catboost import CatBoostRegressor

cb=CatBoostRegressor()


model=cb.fit(X_train,y_train)

y_pred=model.predict(X_test)


##Transform back to its original form....##

y_pred=np.array(y_pred) ## convertd it into numpy array...##

y_pred1=y_pred.reshape(1,-1)### 2d array,as required...###

y_pred_transformed=scaler.inverse_transform(y_pred1)

single_d=y_pred_transformed.reshape(-1)### again converted into 1D array..##


df = pd.DataFrame({'id':data1['id'] ,'sales': single_d})

forecastes_value=df.to_csv("C:/Users/Sayan Mondal/Desktop/demand-forecasting-kernels-only/submission.csv", index=False)





















