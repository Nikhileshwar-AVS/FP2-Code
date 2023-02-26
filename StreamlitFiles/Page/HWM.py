#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

def HWM(coal_data):
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

    st.write("Model Starts here")
    hw_model = ExponentialSmoothing(train["Coal_Con"],
                                  trend    ="mul",
                                  seasonal = "mul", 
                                  seasonal_periods=12).fit()

    hw_fitted = hw_model.fittedvalues

    hw_resid = hw_model.resid

#Adding the mean of the residuals to correct the bias.
    py_hw = hw_model.forecast(len(test["Coal_Con"]))+np.mean(hw_resid)
    py_hw=pd.DataFrame(py_hw)
    py_hw.columns=['py_hw']
    py_hw=py_hw.reset_index()
    py_hw=py_hw.drop('index',axis=1)
    predictions1=test.copy()
    predictions1=predictions1.reset_index()
    py_hw2 = pd.concat([predictions1,py_hw],axis=1)
    py_hw2=py_hw2.set_index('Date')

    st.write(py_hw2)

    train["Coal_Con"].plot(figsize=(12,8), style="--", color="gray", legend=True, label="Train")
    hw_fitted.plot(color="b", legend=True, label="HW_Fitted")
    py_hw2["Coal_Con"].plot(style="--",color="r", legend=True, label="Test")
    py_hw2["py_hw"].plot(color="b", legend=True, label="HW_Forecast")
    st.pyplot(plt)
    
    #MAE
    n=mean_absolute_error(py_hw2.Coal_Con, py_hw2["py_hw"])
    st.write("Mean Absolute error is",n)

#MAPE
    k=np.mean(np.abs((py_hw2.Coal_Con - py_hw2["py_hw"]) / py_hw2.Coal_Con)) * 100
    st.write("MAPE is",k)


#RMSE
    r=math.sqrt(mean_squared_error(py_hw2.Coal_Con, py_hw2["py_hw"]))
    st.write("RMSE is",r)

