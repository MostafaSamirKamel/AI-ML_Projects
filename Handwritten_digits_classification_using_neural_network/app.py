import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Load the trained model (make sure the model file is in the same directory)
model = keras.models.load_model('model.h5')

# Streamlit interface for uploading an image
st.title('Handwritten Digit Recognition')
st.write('Upload an image of a digit (0-9) for prediction.')

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image for prediction (resize to 28x28 and convert to grayscale)
    image = image.convert('L')  # Convert image to grayscale (if not already)
    image = image.resize((28, 28))  # Resize the image to 28x28 pixels
    image_array = np.array(image)  # Convert the image to a numpy array

    # Normalize the image
    image_array = image_array / 255.0  # Normalize pixel values to [0, 1]

    # Flatten the image to match the model input shape (784,)
    image_array = image_array.reshape(1, 28 * 28)  # Flatten to 784 length vector

    # Prediction button
    if st.button('Predict Digit'):
        with st.spinner('Making prediction...'):
            # Make prediction
            prediction = model.predict(image_array)
            predicted_label = np.argmax(prediction, axis=1)[0]  # Get the label with the highest probability

            # Display the prediction result
            st.write(f"The predicted digit is: **{predicted_label}**")

            # Display confidence scores for each digit
            st.write("Confidence Scores (for digits 0-9):")
            st.bar_chart(prediction[0])
