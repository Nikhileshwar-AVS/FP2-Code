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

def SARIMA(coal_data,predict_steps):
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
    seasonal_period=12
    
    st.write("Model Starts here")
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2],seasonal_period) for x in list(itertools.product(p, d, q))]
    
    mini = float('+inf')
    
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
               
                
                if results.aic < mini:
                    mini = results.aic
                    param_mini = param
                    param_seasonal_mini = param_seasonal

#                 print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    st.write('Order:',param_mini)
    st.write('Seasonal Order:',param_seasonal)
    st.write('The set of parameters with the minimum AIC is: SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini, mini))
    
    order=param_mini
    seasonal_order=param_seasonal_mini
    pred_date='2018-12-01'
    mod = sm.tsa.statespace.SARIMAX(y,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

    results = mod.fit()
    st.write(results.summary())
    
    
    # The dynamic=False argument ensures that we produce one-step ahead forecasts, 
    # meaning that forecasts at each point are generated using the full history up to that point.
    pred = results.get_prediction(start=pd.to_datetime(pred_date), dynamic=False)
    pred_ci = pred.conf_int()
    y_forecasted = pred.predicted_mean
    k=mean_absolute_error(y_forecasted, y_to_val)
    n=np.mean(np.abs((y_to_val - y_forecasted) / y_to_val)) * 100
    mse = ((y_forecasted - y_to_val) ** 2).mean()
    st.write("Mean Absolute error :",k)
    st.write("MAPE is :",n)
    st.write('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = False {}'.format(seasonal_period,round(np.sqrt(mse), 2)))

    ax = y.plot(label='observed')
    y_forecasted.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    plt.legend()
    plt.show()
    st.pyplot(plt)

    # A better representation of our true predictive power can be obtained using dynamic forecasts. 
    # In this case, we only use information from the time series up to a certain point, 
    # and after that, forecasts are generated using values from previous forecasted time points.
    pred_dynamic = results.get_prediction(start=pd.to_datetime(pred_date), dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int()
    y_forecasted_dynamic = pred_dynamic.predicted_mean
    k_dynamic=mean_absolute_error(y_forecasted_dynamic, y_to_val)
    n_dynamic=np.mean(np.abs((y_to_val - y_forecasted_dynamic) / y_to_val)) * 100
    mse_dynamic = ((y_forecasted_dynamic - y_to_val) ** 2).mean()
    st.write("Mean Absolute error :",k_dynamic)
    st.write("MAPE is :",n_dynamic)
    st.write('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = True {}'.format(seasonal_period,round(np.sqrt(mse_dynamic), 2)))

    ax = y.plot(label='observed')
    y_forecasted_dynamic.plot(label='Dynamic Forecast', ax=ax,figsize=(14, 7))
    ax.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')

    plt.legend()
    plt.show()
    st.pyplot(plt)

    
    pred_uc = results.get_forecast(steps=predict_steps)
    pred_ci = pred_uc.conf_int()
    pci = pred_ci.reset_index()
    pci.columns = ['Date','Lower Bound','Upper Bound']
    
    
    pm = pred_uc.predicted_mean.reset_index()
    pm.columns = ['Date','Predicted_Mean']
    
    pci2=pci.set_index('Date')
    pm2=pm.set_index('Date')
    ax = y.plot(label='observed', figsize=(14, 7))
    pm2.plot(ax=ax, label='Forecast')
    ax.fill_between(pci2.index,
                    pci2.iloc[:, 0],
                    pci2.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel(y.name)

    plt.legend()
    plt.show()
    st.pyplot(plt)

    final_table = pm.join(pci.set_index('Date'), on='Date')
    final_table=final_table.set_index('Date')
    st.write(final_table)

