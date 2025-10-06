import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Load the pre-trained model
model = tf.keras.models.load_model('./model/fashion_mnist_cnn_model.h5')

# Defuene class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Function to preprocess the image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28)).convert('L')  # Resize and convert to grayscale
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app
st.title("Fashion MNIST Image Classification")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        resized_image = image.resize((100, 100))
        st.image(resized_image, caption='Uploaded Image', use_container_width=True)
    with col2:
        if st.button('Predict'):
            # Preprocess the image
            img_array = preprocess_image(uploaded_file)

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            predicted_class = class_names[np.argmax(predictions)]

            # Display the prediction
            st.write(f"Predicted Class: **{predicted_class}**")