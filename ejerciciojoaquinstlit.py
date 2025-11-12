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

def predict_geometric_next():
    # Inputs for geometric progression: 1, 2, 4, 8 (2^0, 2^1, 2^2, 2^3)
    x_geo = np.array([[2**i] for i in range(4)], dtype=np.float32)
    data_geo = {'instances': x_geo.tolist()}
    response_geo = requests.post(url, json=data_geo)
    # Response is assumed to be like {"predictions": [[..], [..], [..], [..]]}
    if response_geo.status_code == 200:
        preds = response_geo.json().get("predictions", [])
        if preds:
            # Predict the next one: x = 2^4 = 16
            x_next = np.array([[16]], dtype=np.float32)
            data_next = {'instances': x_next.tolist()}
            response_next = requests.post(url, json=data_next)
            if response_next.status_code == 200:
                pred_next = response_next.json().get("predictions", [[None]])[0][0]
                return preds, pred_next
            else:
                return preds, None
        else:
            return None, None
    else:
        return None, None

st.header("Predicción con modelo lineal (secuencia aritmética)")
data = st.text_input('0, 1, 2')
btnPredict = st.button('Predict')

if btnPredict:
    prediction = predict()
    st.write(prediction)
    st.write(prediction.text)

# NUEVA SECCIÓN - SERIE GEOMÉTRICA (como se desea)
st.header("Predicción de serie geométrica")
st.markdown("Serie: **1, 2, 4, 8, 16, ?**")

user_input = st.text_input("¿Cuál es el siguiente número en la serie? (Ingresa tu predicción)", "")
btnGeoPredict = st.button('Verificar predicción')

if btnGeoPredict:
    preds, pred_next = predict_geometric_next()
    if preds is not None and pred_next is not None:
        try:
            user_answer = float(user_input)
            # Accept a small tolerance since ML output may not be exactly 16
            if int(user_answer) == int(pred_next):
                st.success(f"¡Correcto! El modelo predice: {int(pred_next)}")
            else:
                st.error(f"Incorrecto. El modelo predice: {int(pred_next)}, tú pusiste: {user_answer}")
        except ValueError:
            st.warning("Por favor, ingresa un número válido.")
        
        # (opcional) Mostrar las predicciones de los primeros términos
        st.markdown("Predicciones del modelo para la serie dada:")
        for i, pred in enumerate(preds):
            st.write(f"{2**i} → {pred[0]:.2f}")
    else:
        st.error("Error obteniendo la predicción del modelo.")
