import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
import numpy as np
import os

# Obtener la ruta del directorio actual
current_directory = os.path.dirname(os.path.realpath(__file__))


# Combinar la ruta del directorio actual con el nombre del modelo
model_path = os.path.join(current_directory, 'resnet50_model.h5')

# Cargar el modelo previamente entrenado
model = load_model(model_path)

st.title('Detector de Insectos con ResNet50')

# Cargar la imagen de entrada
uploaded_file = st.file_uploader("Cargar una imagen", type="jpg")

if uploaded_file is not None:
    # Preprocesar la imagen para que coincida con el formato que ResNet50 espera
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Realizar la inferencia
    predictions = model.predict(img_array)

    # Mostrar los resultados
    st.image(img, caption='Imagen de entrada', use_column_width=True)
    st.write('Predicciones:')
    st.write(predictions)
