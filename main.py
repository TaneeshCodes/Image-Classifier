import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os


MODEL_PATH = os.path.join("models", "dogcatmodel.h5") 
IMAGE_SIZE = (256, 256) 
CLASS_NAMES = ["Cat", "Dog"] 


@st.cache_resource 
def load_model():
    """Loads the pre-trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image):
    """
    Resizes the image and normalizes pixel values.
    Adjust preprocessing based on how your model was trained.
    """
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image)
  
    if image_array.ndim == 2: 
        image_array = np.stack([image_array, image_array, image_array], axis=-1)
    elif image_array.shape[2] == 4: 
        image_array = image_array[:, :, :3] 

    image_array = image_array / 255.0 

    image_array = np.expand_dims(image_array, axis=0)
    return image_array
#

st.set_page_config(page_title="Dog vs Cat Classifier", layout="centered")

st.title("🐾 Dog vs Cat Image Classifier 🐱")
st.write("Upload an image and let the CNN model tell you if it's a dog or a cat!")

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    

    if predictions[0][0] > 0.5: 
        predicted_class = CLASS_NAMES[1] # Dog
        confidence = predictions[0][0]
    else:
        predicted_class = CLASS_NAMES[0] # Cat
        confidence = 1 - predictions[0][0]

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence:.2f}**")



st.sidebar.header("About")
st.sidebar.info(
    "This application uses a Convolutional Neural Network (CNN) model "
    "to classify uploaded images as either a dog or a cat. "
    "The model `dogcatmodel.h5` is located in the `models` directory."
)

st.sidebar.text("Developed with Streamlit and TensorFlow")