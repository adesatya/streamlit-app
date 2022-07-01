# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 00:39:51 2022

@author: ade.satya
"""

import streamlit as st
import joblib
import numpy as np

def main():
    with open('model.pkl', 'rb') as model_file:
        model = joblib.load(model_file)
    
    st.title('Belajar front end prediction')
    st.subheader('Silahkan masukan variable')
    
    #pw = st.number_input('Petal Width',  value=5.2,min_value=0.0, max_value=8.0, step=0.1)
    #pl = st.number_input('Petal length',  value=3.2,min_value=0.0, max_value=8.0, step=0.1)
    #sw = st.number_input('Sepal Width',  value=4.2,min_value=0.0, max_value=8.0, step=0.1)
    #sl = st.number_input('Sepal length',  value=1.2,min_value=0.0, max_value=8.0, step=0.1)
    
    pw = st.slider('Petal Width',  value=5.2,min_value=0.0, max_value=8.0, step=0.1)
    pl = st.slider('Petal length',  value=3.2,min_value=0.0, max_value=8.0, step=0.1)
    sw = st.slider('Sepal Width',  value=4.2,min_value=0.0, max_value=8.0, step=0.1)
    sl = st.slider('Sepal length',  value=1.2,min_value=0.0, max_value=8.0, step=0.1)
    
    if st.button("Click Here to Classify"):
        data_input = np.array([sl,sw,pl,pw]).reshape(1, -1)
        prediction = model.predict(data_input)[0]
        st.sidebar.subheader('Hasil prediksi adalah:')
        if prediction == 1:
            st.sidebar.write('setosa')
        elif prediction == 2:
            st.sidebar.write('versicolor')
        elif prediction == 3:
            st.sidebar.write('virginica')
        
    
if __name__ == '__main__':
    main()
