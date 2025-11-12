## Por - Joaquin Saldarriaga
## Modificacion de 'predict' orgiginal (0, 1, 2) - clase 11 de nov. 2025
## Modificacion: se añadió 'predict_geometric' para predecir la siguiente en serie geométrica
import streamlit as st
import requests
import numpy as np

url = 'https://tensorflow-linear-model-fvap.onrender.com/v1/models/linear-model:predict'

def predict():
    x = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ], dtype=np.float32)
    
    data = {'instances': x.tolist()}
    response = requests.post(url, json=data)
    
    print(response.text)
    return response

def predict_geometric():
    """
    Send 2^0, 2^1, 2^2, 2^3 etc. as geometric progression for prediction.
    """
    x_geo = np.array([
        [2**0],
        [2**1],
        [2**2],
        [2**3]
    ], dtype=np.float32)

    data_geo = {'instances': x_geo.tolist()}
    response_geo = requests.post(url, json=data_geo)
    
    print(response_geo.text)
    return response_geo

st.header("Predicción con modelo lineal (secuencia aritmética)")
data = st.text_input('0, 1, 2')
btnPredict = st.button('Predict')

if btnPredict:
    prediction = predict()
    st.write(prediction)
    st.write(prediction.text)

st.header("Predicción de serie geométrica (2^0, 2^1, 2^2, ...)")
btnGeoPredict = st.button('Predict the next number in the geometric series')

if btnGeoPredict:
    prediction_geo = predict_geometric()
    st.write(prediction_geo)
    st.write(prediction_geo.text)
