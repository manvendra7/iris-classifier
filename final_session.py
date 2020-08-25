# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:21:40 2020

@author: sa
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle 

with open('iris_classifier.pickle','rb') as f:
    classifier = pickle.load(f)
    

def main():
    st.title('Iris Species classifier')
    SepalLength = st.text_input('SepalLength')
    Sepalwidth = st.text_input('SepalWidth')
    PetalLength = st.text_input('PetalLength')
    PetalWidth = st.text_input('PetalWidth')
    result = ""
    if st.button('Predict'): 
        result = classifier.predict([[SepalLength,Sepalwidth,PetalLength,PetalWidth]])
        if result == 0:
            species = 'Iris-Setosa'
            st.success('The species of iris is {}'.format(species))
        elif result == 1:
            species = 'Iris-Versicolor'
            st.success('The species of iris is {}'.format(species))
        elif result == 2:
            species = 'Iris-Virginica'
            st.success('The species of iris is {}'.format(species))
    
if __name__== '__main__' :
    main()
    
    
    
    
    
    

