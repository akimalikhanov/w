import streamlit as st
import sys
sys.path.insert(1, '../src')
import config
import requests
from utils import *
import pandas as pd
import json


st.title('Credit Risk Modelling')
uploaded_file=st.file_uploader("Choose a file")
cutoff=st.slider('Choose your cutoff value (minimum probability that would be considered a "default")', 0., 1., 0.5)
submit=st.button('Submit')
if submit and uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    data=read_sample(df)
    response=requests.post('http://127.0.0.1:8000/predict', json=data)
    pred=json.loads(response.text)['prediction']
    st.write(f'Probability of Default is {pred:.3f}')
    if pred>cutoff:
        st.error(f'The applicant is likely to default!')
    else:
        st.success(f'The applicant is not likely to default:)')