import streamlit as st
import pandas as pd
import pickle
import sklearn
from sklearn import preprocessing

#load model and scalers
loaded_model = pickle.load(open("sales-advertising-model.h5","rb"))
scaler_features = pickle.load(open("feature-scaler.pkl","rb"))
scaler_sales = pickle.load(open("sales-scaler.pkl","rb"))

