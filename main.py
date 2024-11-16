import streamlit as st
import panda as pd
import plotly.express as px

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from scipy.special import inv_boxcox

def run():
  st.title("🎈 project cancer risk")

  st.subheader('Raw Data')
