import streamlit as st
import pandas as pd
import pickle
import sklearn
from sklearn import preprocessing

loaded_model = pickle.load(open("sales-advertising-model.h5","rb"))
scaler_features = pickle.load(open("features-scaler.pkl","rb"))
scaler_sales = pickle.load(open("sales-scaler.pkl","rb"))

st.title("Sales Prediction App")
st.write("This app predicts the sales based on three advertising channel features")
