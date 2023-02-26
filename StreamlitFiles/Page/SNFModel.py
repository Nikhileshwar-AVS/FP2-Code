#!/usr/bin/env python
# coding: utf-8

# In[2]:


# determining the parameters
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#libaries for time series
import statsmodels
from statsmodels.tools.eval_measures import rmse
import seaborn as sns
import statsmodels.api as sm
import itertools
from statsmodels.tsa.arima_model import ARIMA, ARMA
import warnings
warnings.filterwarnings("ignore")
import sklearn
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing 
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.metrics import accuracy_score
import math 
import itertools
    
    

def SNFM(coal_data):
    st.write("Starting cleaning & Pre-processing")
    coal_data=coal_data.rename(columns={'Series ID: ELEC.CONS_TOT.COW-AK-96.M thousand tons':'Coal_Con'})
    mean_values = (coal_data['Coal_Con'].shift() + coal_data['Coal_Con'].shift(-1)) / 2
    # Replace missing values with the mean of the preceding and following values
    coal_data['Coal_Con'].fillna(mean_values, inplace=True)
    coal_data['D'] = coal_data['Month'].str.slice(0, 2)
    coal_data['M'] = coal_data['Month'].str.slice(3, 5)
    coal_data['Y'] = coal_data['Month'].str.slice(6, 10)
    coal_data['Date1'] = coal_data['Y']+'-'+coal_data['M']+'-'+coal_data['D']
    coal_data['Date'] = pd.to_datetime(coal_data['Date1'])
    coal_data3=coal_data.sort_values(['Date'],ascending=True)
    coal_data4=coal_data3[['Date','Coal_Con']]
    df = coal_data4.set_index("Date")
    y=df['Coal_Con']
    y_to_val = y['2018-12-01':] # dataset to train
    y_to_train = y[:'2018-12-01'] # last X months for test  
    #predict_date = len(y) - len(y[:'Q4 2017']-1) 
    train=pd.DataFrame(y_to_train)
    test=pd.DataFrame(y_to_val)

    train_series=train["Coal_Con"]
    seasonal_periods=12
    forecast_horizon=len(train)

    last_season=train_series.iloc[-seasonal_periods:]
        
    reps=int(np.ceil(forecast_horizon/seasonal_periods))
        
    fcarray=np.tile(last_season,reps)
        
    fcast=pd.Series(fcarray[:forecast_horizon])
        
    fitted = train_series.shift(seasonal_periods)
    py_snaive_fit = fitted
    #forecast
    py_snaive = fcast
    

#Residuals
    py_snaive_resid = (train["Coal_Con"] - py_snaive_fit).dropna()



    predictions=train.copy()
    predictions["py_snaive"] = py_snaive.values 

    st.write(predictions)

# Charting the output
    pd.plotting.register_matplotlib_converters()
    train["Coal_Con"].plot(figsize=(12,8))#, style="--", color="gray", legend=True, label="Train")
    py_snaive_fit.plot(color="b", legend=True, label="SNaive_Fitted")
    predictions["Coal_Con"].plot(style="--",color="r", legend=True, label="Test")
    st.pyplot(plt)

#MAE
    n=mean_absolute_error(predictions["Coal_Con"], predictions["py_snaive"])
    st.write("Mean Absolute error is",n)

#MAPE
    k=np.mean(np.abs((predictions["Coal_Con"] - predictions["py_snaive"]) / predictions["Coal_Con"])) * 100
    st.write("MAPE is",k)

#RMSE
    r=math.sqrt(mean_squared_error(predictions["Coal_Con"], predictions["py_snaive"]))
    st.write("RMSE is",r)


# In[ ]:




