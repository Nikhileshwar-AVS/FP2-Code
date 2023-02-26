#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Preprocessing(coal_data):
    st.write("Doing necessary cleaning")
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
    st.write("Done")
    st.write(coal_data4)
    df = coal_data4.set_index("Date")
    df['Coal_Con'].plot(style="-")
    plt.title("Time Series Plot")
    plt.ylabel("Time")
    plt.ylabel("Coal Consumption")
    st.pyplot(plt)


# In[ ]:




