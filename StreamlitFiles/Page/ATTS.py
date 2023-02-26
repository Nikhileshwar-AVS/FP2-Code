#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import autots
from autots import AutoTS, load_monthly
long = False

def ATTS(coal_data,forecast_length):
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
    model = AutoTS(
    forecast_length=forecast_length,
    frequency='infer',
    prediction_interval=0.9,
    ensemble=None,
    model_list="fast",  # "superfast", "default", "fast_parallel"
    transformer_list="fast",  # "superfast",
    drop_most_recent=1,
    max_generations=4,
    num_validations=2,
    validation_method="backwards"
    )
    model = model.fit(
    df,
    date_col='Date' if long else None,
    value_col='Coal_Con' if long else None
    )
    prediction = model.predict()
    # plot a sample
    prediction.plot(model.df_wide_numeric,
                    series=model.df_wide_numeric.columns[0],
                    start_date="2001-01-01")
    st.pyplot(plt)
    st.write("The details of the best model")
    st.write(model)
    # point forecasts dataframe
    forecasts_df = prediction.forecast
    # upper and lower forecasts
    forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

    # accuracy of all tried model results
    model_results = model.results()
    # and aggregated from cross validation
    validation_results = model.results("validation")
    st.write(validation_results)
    

