#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from Page.Preprocessing import *
from Page.SNFModel import *
from Page.HWM import*
from Page.ETS import*
from Page.SARIMA import*
from Page.ATTS import*
import random

def save_uploadedfile(uploadedfile):
    st.session_state['key'] = random.randint(0,99999)
    with open(os.path.join("temp"+str(st.session_state['key'])+".csv"),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{}".format(uploadedfile.name))

def main_page():
    st.markdown("## Data Upload")
    func_check()


def page2():
    st.markdown("## Cleaning & Pre-processing")
    if "key" in st.session_state:
        if os.path.isfile("temp"+str(st.session_state['key'])+".csv"):
            if os.path.isfile("temp_cleaned"+str(st.session_state['key'])+".csv"):
                data = pd.read_csv("temp_cleaned"+str(st.session_state['key'])+".csv",encoding='cp1252')
                st.write(data)
            else:
                data = pd.read_csv("temp"+str(st.session_state['key'])+".csv",encoding='cp1252')
                Preprocessing(data)
        else:
            st.write("Please upload data to proceed further")
    else:
        st.write("Please upload data to proceed further")
        

def page3():
    st.markdown("## Seasonal Naive Forecast model")
    if "key" in st.session_state:
        if os.path.isfile("temp"+str(st.session_state['key'])+".csv"):
            if os.path.isfile("temp_cleaned"+str(st.session_state['key'])+".csv"):
                data = pd.read_csv("temp_cleaned"+str(st.session_state['key'])+".csv",encoding='cp1252')
                st.write(data)
            else:
                data = pd.read_csv("temp"+str(st.session_state['key'])+".csv",encoding='cp1252')
                SNFM(data)
        else:
            st.write("Please upload data to proceed further")
    else:
        st.write("Please upload data to proceed further")
        

def page4():
    st.markdown("## Holt Winter Method")
    if "key" in st.session_state:
        if os.path.isfile("temp"+str(st.session_state['key'])+".csv"):
            if os.path.isfile("temp_cleaned"+str(st.session_state['key'])+".csv"):
                data = pd.read_csv("temp_cleaned"+str(st.session_state['key'])+".csv",encoding='cp1252')
                st.write(data)
            else:
                data = pd.read_csv("temp"+str(st.session_state['key'])+".csv",encoding='cp1252')
                HWM(data)
        else:
            st.write("Please upload data to proceed further")
    else:
        st.write("Please upload data to proceed further")


def page5():
    st.markdown("## ETS Model")
    if "key" in st.session_state:
        if os.path.isfile("temp"+str(st.session_state['key'])+".csv"):
            if os.path.isfile("temp_cleaned"+str(st.session_state['key'])+".csv"):
                data = pd.read_csv("temp_cleaned"+str(st.session_state['key'])+".csv",encoding='cp1252')
                st.write(data)
            else:
                data = pd.read_csv("temp"+str(st.session_state['key'])+".csv",encoding='cp1252')
                ETS(data)
        else:
            st.write("Please upload data to proceed further")
    else:
        st.write("Please upload data to proceed further")

        


def page6():
    st.markdown("## SARIMA")
    if "key" in st.session_state:
        if os.path.isfile("temp"+str(st.session_state['key'])+".csv"):
            if os.path.isfile("temp_cleaned"+str(st.session_state['key'])+".csv"):
                data = pd.read_csv("temp_cleaned"+str(st.session_state['key'])+".csv",encoding='cp1252')
                st.write(data)
            else:
                data = pd.read_csv("temp"+str(st.session_state['key'])+".csv",encoding='cp1252')
                SARIMA(data,30)
        else:
            st.write("Please upload data to proceed further")
    else:
        st.write("Please upload data to proceed further")
        

def page7():
    st.markdown("## AutoTS")
    if "key" in st.session_state:
        if os.path.isfile("temp"+str(st.session_state['key'])+".csv"):
            if os.path.isfile("temp_cleaned"+str(st.session_state['key'])+".csv"):
                data = pd.read_csv("temp_cleaned"+str(st.session_state['key'])+".csv",encoding='cp1252')
                st.write(data)
            else:
                data = pd.read_csv("temp"+str(st.session_state['key'])+".csv",encoding='cp1252')
                ATTS(data,30)
        else:
            st.write("Please upload data to proceed further")
    else:
        st.write("Please upload data to proceed further")
        
        
page_names_to_funcs = {
    "Upload": main_page,
    "Preprocessing": page2,
    "Seasonal Naive Forecast model":page3,
    "Holt Winter Method":page4,
    "ETS Model":page5,
    "SARIMA":page6,
    "AutoTS":page7
}

def func_check():
    
    
    st.write('')
    with open('Total_consumption_coal_Alaska_all_commercial_(total)_monthly.csv', 'rb') as f:
        st.download_button(
         label="Download data for this app",
         data = f,
         file_name='Total_consumption_coal_Alaska_all_commercial_(total)_monthly.csv',
         mime='text/csv')
    
    uploaded_file = st.file_uploader("Choose a file",type=["csv"])
    
    if uploaded_file is not None:
         # To read file as bytes:
         try:
             data = pd.read_csv(uploaded_file,encoding='cp1252')
         except Exception as e:
             st.write("Error",e)
         finally:
             save_uploadedfile(uploaded_file)
             st.dataframe(data)
    else:
        if "key" in st.session_state:
            if os.path.isfile("temp"+str(st.session_state['key'])+".csv"):
                os.remove("temp"+str(st.session_state['key'])+".csv")
            if os.path.isfile("temp_cleaned"+str(st.session_state['key'])+".csv"):
                os.remove("temp_cleaned"+str(st.session_state['key'])+".csv")
    
    
  
def main():
    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    with st.spinner('Wait for it...'):
        time.sleep(0.5)
        page_names_to_funcs[selected_page]()
        
if __name__ == '__main__':
    main()


# In[ ]:




