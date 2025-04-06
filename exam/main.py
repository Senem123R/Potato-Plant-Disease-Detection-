import streamlit as st
import tensorflow as tf
import numpy as np

# TensorFlow model prediction
def model_prediction(test_image):
    model_y = tf.keras.models.load_model("fixed1_model.keras", compile=False)

    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Expand dims for batch
    predictions = model_y.predict(input_arr)
    result_index = np.argmax(predictions[0])
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select the app mode", ("Home", "Predict"))

if app_mode == "Home":
    st.title("Welcome to the Home Page!")
    st.write("This is the home page of your Streamlit app.")

elif app_mode == "Predict":
    st.header("Prediction Page")
    test_image = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
    
    if test_image and st.button("Show Image"):
        st.image(test_image, caption='Uploaded Image.', use_container_width=True)

    if test_image and st.button("Predict"):
        st.write("Predicting...")
        result_index = model_prediction(test_image)
        if result_index is not None:
            class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
            st.success(f"Model prediction: {class_name[result_index]}")
