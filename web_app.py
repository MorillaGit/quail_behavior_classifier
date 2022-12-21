from src import utils_module
from src import preprocessing_module
from src import utils_module
from src import deep_learning_module
import importlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
import streamlit as st
import os
import time




# initial container streamlit
st.set_page_config(layout="wide", page_title="Quail Research Tools")


def main():

    st.title("Quail Research Tools")

    st.write("## Quail Time Series Analysis Behavior Classifier")
    st.write(""" :bird: Quail are small birds that are commonly used in research studies due to their unique behavior and cognitive abilities. However, tracking and analyzing the behavior of quail can be challenging, especially when using traditional methods such as manual observation or video tracking. In this project, we present an open source solution for the detection and classification of quail behavior using time series data from accelerometers. By leveraging the power of machine learning and time series analysis, our approach enables the automatic and accurate tracking of quail behavior in a variety of research settings. With this tool, researchers can easily experiment with different machine learning models and feature engineering techniques to gain valuable insights into the behavior and cognitive abilities of quail. In addition, this tool can be used to test the effectiveness of different interventions or treatments on quail behavior.. This code is open source and available 
    [here](https://github.com/MorillaGit/quail_behavior_classifier) on GitHub. 
    Special thanks to the , [Jackelyn Melissa Kembro](https://www.researchgate.net/profile/Jackelyn-Kembro) and [Pedro A. Pury](https://scholar.google.com/citations?user=HPlHHewAAAAJ&hl=es) :grin:""")
    
    st.sidebar.write("## Research Tools :gear:")

    menu = ["Home", "load","preprocessing", "processing", "model", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
    elif choice == "load":
        st.subheader("Load")
    elif choice == "preprocessing":
        st.subheader("Preprocessing")
    elif choice == "processing":
        st.subheader("Processing")
    elif choice == "model":
        st.subheader("Model")
    elif choice == "About":
        st.subheader("About")




if __name__ == '__main__':
    main()