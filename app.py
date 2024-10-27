import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model('/content/drive/My Drive/bird_species_classifier.keras')

# Define the class labels
class_labels = os.listdir('/content/drive/My Drive/Bird/train_data/train_data')  # Adjust the path as necessary

def preprocess_image(img):
    """Preprocess the uploaded image for prediction."""
    img = img.resize((150, 150))  # Resize to match model input
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale pixel values
    return img_array

def predict_image(img):
    """Predict the class of an uploaded image."""
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    return class_labels[predicted_class[0]], prediction[0][predicted_class[0]]  # Return class name and confidence

# Streamlit app layout
st.title("🦜 Bird Species Classifier")
st.markdown("""
    <style>
        .title {
            color: #4CAF50;
            font-size: 40px;
            text-align: center;
        }
        .description {
            color: #555555;
            font-size: 18px;
            text-align: center;
        }
        .predicted {
            color: #FF5722;
            font-size: 24px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.write('<div class="description">Upload an image of a bird to classify its species.</div>', unsafe_allow_html=True)

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = image.load_img(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    predicted_class, confidence = predict_image(img)

    # Display prediction results
    st.write(f'<div class="predicted">Predicted: {predicted_class} (Confidence: {confidence:.2f})</div>', unsafe_allow_html=True)
