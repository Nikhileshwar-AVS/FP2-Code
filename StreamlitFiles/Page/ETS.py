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

def ETS(coal_data):
    st.write("Starting cleaning & Pre-processing")
    coal_data=coal_data.rename(columns={'Series ID: ELEC.CONS_TOT.COW-AK-96.M thousand tons':'Coal_Con'})
    mean_values = (coal_data['Coal_Con'].shift() + coal_data['Coal_Con'].shift(-1)) / 2
    coal_data['Coal_Con'].fillna(mean_values, inplace=True)
    coal_data['D'] = coal_data['Month'].str.slice(0, 2)
    coal_data['M'] = coal_data['Month'].str.slice(3, 5)
    coal_data['Y'] = coal_data['Month'].str.slice(6, 10)
    coal_data['Date1'] = coal_data['Y']+'-'+coal_data['M']+'-'+coal_data['D']
    coal_data['Date'] = pd.to_datetime(coal_data['Date1'])
    coal_data3=coal_data.sort_values(['Date'],ascending=True)
    coal_data4=coal_data3[['Month','Coal_Con']]
    df = coal_data4.set_index("Month")
    y=df['Coal_Con']
    y_to_val = y['01-12-2018':] # dataset to train
    y_to_train = y[:'01-12-2018'] # last X months for test  
    #predict_date = len(y) - len(y[:'Q4 2017']-1) 
    train=pd.DataFrame(y_to_train)
    test=pd.DataFrame(y_to_val)
    
    
    st.write("Model Starts here")
    ets_AAdA=sm.tsa.statespace.ExponentialSmoothing(train["Coal_Con"],
                                           trend=True, 
                                           initialization_method= 'concentrated', 
                                           seasonal=12, 
                                           damped_trend=True).fit()

    fc_AAdA=ets_AAdA.forecast(len(test))

    fc_AAdA=pd.DataFrame(fc_AAdA)
    fc_AAdA.columns=['fc_AAdA']
    fc_AAdA=fc_AAdA.reset_index()
    fc_AAdA=fc_AAdA.drop('index',axis=1)
    predictions2=test.copy()
    predictions2=predictions2.reset_index()
    fc_AAdA2 = pd.concat([predictions2,fc_AAdA],axis=1)
    fc_AAdA2=fc_AAdA2.set_index('Month')
    fc_AAdA2=fc_AAdA2.reset_index()
    fc_AAdA2['D'] = fc_AAdA2['Month'].str.slice(0, 2)
    fc_AAdA2['M'] = fc_AAdA2['Month'].str.slice(3, 5)
    fc_AAdA2['Y'] = fc_AAdA2['Month'].str.slice(6, 10)
    fc_AAdA2['Date1'] = fc_AAdA2['Y']+'-'+fc_AAdA2['M']+'-'+fc_AAdA2['D']
    fc_AAdA2['Date'] = pd.to_datetime(fc_AAdA2['Date1'])
    fc_AAdA2=fc_AAdA2[['Date','Coal_Con','fc_AAdA']]
    fc_AAdA2=fc_AAdA2.set_index('Date')
    
    ets_AAdA2=pd.DataFrame(ets_AAdA.fittedvalues)
    ets_AAdA2=ets_AAdA2.reset_index()
    ets_AAdA2['D'] = ets_AAdA2['Month'].str.slice(0, 2)
    ets_AAdA2['M'] = ets_AAdA2['Month'].str.slice(3, 5)
    ets_AAdA2['Y'] = ets_AAdA2['Month'].str.slice(6, 10)
    ets_AAdA2['Date1'] = ets_AAdA2['Y']+'-'+ets_AAdA2['M']+'-'+ets_AAdA2['D']
    ets_AAdA2['Date'] = pd.to_datetime(ets_AAdA2['Date1'])
    ets_AAdA2=ets_AAdA2[['Date',0]]
    ets_AAdA2=ets_AAdA2.set_index('Date')
    train2=train.reset_index()
    train2['D'] = train2['Month'].str.slice(0, 2)
    train2['M'] = train2['Month'].str.slice(3, 5)
    train2['Y'] = train2['Month'].str.slice(6, 10)
    train2['Date1'] = train2['Y']+'-'+train2['M']+'-'+train2['D']
    train2['Date'] = pd.to_datetime(train2['Date1'])
    train2=train2[['Date','Coal_Con']]
    train2=train2.set_index('Date')
    st.write(fc_AAdA2)
    
    train2["Coal_Con"].plot(figsize=(12,8), style="--", color="gray", legend=True, label="Train")
    ets_AAdA2[0].plot(color="b", legend=True, label="AAdA_Fitted")
    fc_AAdA2["Coal_Con"].plot(style="--",color="r", legend=True, label="Test")
    fc_AAdA2["fc_AAdA"].plot(color="b", legend=True, label="AAdA_Forecast")
    st.pyplot(plt)
    
    n=mean_absolute_error(fc_AAdA2.Coal_Con, fc_AAdA2["fc_AAdA"])
    st.write("Mean Absolute error is",n)

        #MAPE
    k=np.mean(np.abs((fc_AAdA2.Coal_Con - fc_AAdA2["fc_AAdA"]) / fc_AAdA2.Coal_Con)) * 100
    st.write("MAPE is",k)

    #RMSE
    r=math.sqrt(mean_squared_error(fc_AAdA2.Coal_Con, fc_AAdA2["fc_AAdA"]))
    st.write("RMSE is",r)

