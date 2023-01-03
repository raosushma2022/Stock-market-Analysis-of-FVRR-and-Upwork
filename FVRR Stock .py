#!/usr/bin/env python
# coding: utf-8

# # FVRR Stock Price Analysis

# In[1]:


#input the necessary libraries


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os


# In[3]:


##Loading the Excel files 
data= pd.read_csv("FVRR.csv",index_col='Date',parse_dates=True)


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


data.describe()


# In[9]:


data['Open'].plot(figsize=(16,6))


# In[10]:


data['Close'].pct_change().plot.hist(bins=50)
plt.xlabel("Adjusted close 1 day percent change")
plt.show()


# From the histogram, we can observe the price change percentage mostly fall between the range of -0.3% to 0.3%.

# In[11]:



import datetime as dt
import pandas as pd
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV


# In[12]:


data.head()


# In[13]:


x = data.iloc[:, 0:5].values
y = data.iloc[:, 4].values


# # x is holding values for the open, high, low, close and adj close columns. yis holding values for the adj-close column. The other columns, including that of volume, were not selected for the process because they will not be needed. Five features were used.

# In[14]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.26,  random_state=0)


# Step 4: Scaling the features

# In[15]:


scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)


# Step 4: Hyperparemater tuning

# For a random forest regression model, the best parameters to consider are:
# 
#     n_estimators — number of trees in the forest
#     max_depth — maximum depth in a tree
#     min_samples_split — minimum number of data points before the sample is split
#     min_samples_leaf — minimum number of leaf nodes that are required to be sampled
#     bootstrap — sampling for data points, true or false
#     random_state — generated random numbers for the random forest.

# In[16]:


model = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_split=2, min_samples_leaf=1, max_depth=10, bootstrap=True)
model.fit(x_train, y_train)
predict = model.predict(x_test)
print(predict)
print(predict.shape)


# In[17]:


grid_rf = {
'n_estimators': [20, 50, 100, 500, 1000],  
'max_depth': np.arange(1, 15, 1),  
'min_samples_split': [2, 10, 9], 
'min_samples_leaf': np.arange(1, 15, 2, dtype=int),  
'bootstrap': [True, False], 
'random_state': [1, 2, 30, 42]
}
rscv = RandomizedSearchCV(estimator=model, param_distributions=grid_rf, cv=3, n_jobs=-1, verbose=2, n_iter=200)
rscv_fit = rscv.fit(x_train, y_train)
best_parameters = rscv_fit.best_params_
print(best_parameters)


# Statistical metrics and performance evaluation

# Root mean square error (RMSE)

# In[18]:


print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, predict), 4))
print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, predict), 4))
print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_test, predict)), 4))
print("(R^2) Score:", round(metrics.r2_score(y_test, predict), 4))
print(f'Train Score : {model.score(x_train, y_train) * 100:.2f}% and Test Score : {model.score(x_test, y_test) * 100:.2f}% using Random Tree Regressor.')
errors = abs(predict - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.') 


# Collect future days from predicted values and plot

# In[19]:


predictions = pd.DataFrame({"Predictions": predict},
index=pd.date_range(start=data.index[-1], periods=len(predict), freq="D"))
predictions.to_csv("Predicted-price-data.csv")
#colllects future days from predicted values
oneyear_df = pd.DataFrame(predictions[:252])
oneyear_df.to_csv("one-year-predictions.csv")
onemonth_df = pd.DataFrame(predictions[:21])
onemonth_df.to_csv("one-month-predictions.csv")
fivedays_df = pd.DataFrame(predictions[:5])
fivedays_df.to_csv("five-days-predictions.csv")


# In[20]:


predictions


# In[21]:


oneyear_df


# In[22]:


onemonth_df


# In[23]:


fivedays_df


# One year prediction :

# In[24]:


oneyear_df_pred = pd.read_csv("one-year-predictions.csv")
#oneyear_df_pred.set_index("Date", inplace=True)
buy_price = min(oneyear_df_pred["Predictions"])
sell_price = max(oneyear_df_pred["Predictions"])
oneyear_buy = oneyear_df_pred.loc[oneyear_df_pred["Predictions"] == buy_price]
oneyear_sell = oneyear_df_pred.loc[oneyear_df_pred["Predictions"] == sell_price]
print("Buy price and date")
print(oneyear_buy)
print("Sell price and date")
print(oneyear_sell)
oneyear_df_pred["Predictions"].plot(figsize=(10, 5), title="Forecast for the next 1 year", color="blue")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


# One month prediction:

# In[25]:


onemonth_df_pred = pd.read_csv("one-month-predictions.csv")
#onemonth_df_pred.set_index("Date", inplace=True)
buy_price = min(onemonth_df_pred["Predictions"])
sell_price = max(onemonth_df_pred["Predictions"])
onemonth_buy = onemonth_df_pred.loc[onemonth_df_pred["Predictions"] == buy_price]
onemonth_sell = onemonth_df_pred.loc[onemonth_df_pred["Predictions"] == sell_price]
print("Buy price and date")
print(onemonth_buy)
print("Sell price and date")
print(onemonth_sell)
onemonth_df_pred["Predictions"].plot(figsize=(10, 5), title="Forecast for the next 1 month", color="blue")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


# Five days prediction :

# In[26]:


fivedays_df_pred = pd.read_csv("five-days-predictions.csv")
#fivedays_df_pred.set_index("Date", inplace=True)
buy_price = min(fivedays_df_pred["Predictions"])
sell_price = max(fivedays_df_pred["Predictions"])
fivedays_buy = fivedays_df_pred.loc[fivedays_df_pred["Predictions"] == buy_price]
fivedays_sell = fivedays_df_pred.loc[fivedays_df_pred["Predictions"] == sell_price]
print("Buy price and date")
print(fivedays_buy)
print("Sell price and date")
print(fivedays_sell)
fivedays_df_pred["Predictions"].plot(figsize=(10, 5), title="Forecast for the next 5 days", color="blue")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


# Conclusion
# 
# There are different approaches to solving these problems. Their performance can differ from mathematical analysis for prediction to sentiment analysis, financial news articles, and expert reviews.Overall, random forest is mostly fast, simple, and flexible, but not without some limitations.

# # STOCK MARKET PREDICTIONS USING LSTM

# In[27]:


##Loading the Excel files 
df1= pd.read_csv("FVRR.csv",index_col='Date',parse_dates=True)


# In[28]:


df1.head()


# In[29]:


df1.shape


# In[30]:


df1=df1['Close']


# In[31]:


df1


# In[32]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[33]:


#Importing Keras Libraries and Packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[34]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[35]:


df1


# In[36]:


len(df1)


# In[37]:


##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[38]:


training_size,test_size


# In[39]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[40]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 10
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[41]:


print(X_train.shape), print(y_train.shape)


# In[42]:




print(X_test.shape), print(ytest.shape)


# In[43]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[44]:


### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[45]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(10,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[46]:


model.summary()


# In[47]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100)


# In[48]:


import tensorflow as tf


# In[49]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[50]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[51]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[52]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[53]:


### Plotting 
# shift train predictions for plotting
look_back=10
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[54]:




len(test_data)


# In[55]:


x_input=test_data[55:].reshape(1,-1)
x_input.shape


# In[56]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[57]:


temp_input


# In[58]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=10
i=0
while(i<30):
    
    if(len(temp_input)>10):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[59]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[60]:




len(df1)


# In[61]:


plt.plot(day_new,scaler.inverse_transform(df1[84:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[62]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[55:])


# In[63]:


df3=scaler.inverse_transform(df3).tolist()


# In[64]:


plt.plot(df3)


# CONCLUSION 
# 
# The stock price of FVRR  is expected to increase in the coming months

# In[ ]:




