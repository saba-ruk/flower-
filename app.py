import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np

# Header
st.header('🌸 Flower Classification CNN Model')

# Class names
flower_names = ['daisy', 'dandelion','rose', 'sunflower', 'tulip']

# Load trained model
model = load_model('Flower_Recog_Model.h5')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])

    outcome = f"The Image belongs to **{flower_names[np.argmax(result)]}** with a score of **{np.max(result)*100:.2f}%**"
    return outcome, result.numpy()


# Upload
uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'png', 'jpeg'])

# Process upload
if uploaded_file is not None:
    # Create a temporary uploads folder if it doesn't exist
    if not os.path.exists('upload'):
        os.makedirs('upload')

    # Save uploaded file to disk
    file_path = os.path.join('upload', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Run prediction
    outcome, probabilities = classify_images(file_path)

    # Show prediction result
    st.markdown(outcome)

    # Layout: Image | Bar Chart side-by-side
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.bar_chart({flower_names[i]: float(prob) for i, prob in enumerate(probabilities)})
