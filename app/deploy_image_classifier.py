from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

@st.cache_data
def load_validation_images():
    data_dir = Path(__file__).parents[1] / "validation_data/"
    data_files = data_dir.glob("*")
    loaded_images = {
        file.stem: tf.keras.preprocessing.image.load_img(file)
        for file in data_files
    }
    return loaded_images


def choose_from_default_images():
    st.markdown(
        """
    By default, we can choose from a selection of images from the validation set 
    to be classified.
    """
    )
    loaded_images = load_validation_images()
    choice = st.selectbox(
        "Select an input image:", list(loaded_images.keys()), index=0
    )
    selected_image = loaded_images[choice]
    st.markdown(
        """
    This is the image that we will classify:
    """
    )
    st.image(selected_image, caption=choice)
    return selected_image

def select_image():
    selected_image = None
    user_input = st.checkbox("Provide a picture?", value=False)
    if user_input:
        st.markdown(
            """
        We can choose to upload an existing image here or use our device's
        camera to take a picture. We will then classify that image.
        """
        )
        user_input_choices = ["Upload a picture", "Use the camera"]
        choice = st.selectbox(
            "Select an input method", user_input_choices, index=0
        )
        if choice == user_input_choices[0]:
            img_file_buffer = st.file_uploader(
                "Upload an image here", type=["jpg", "jpeg", "png"]
            )
        else:
            img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            selected_image = Image.open(img_file_buffer)
    else:
        selected_image = choose_from_default_images()
    return selected_image


def resize_image(img):
    image_array = tf.keras.preprocessing.image.img_to_array(img.resize((128, 128)))
    return np.expand_dims(image_array, axis=0)

@st.cache_resource
def get_trained_image_classifier():
    model_dir = Path(__file__).parents[1] / "trained_model/"
    loaded_model = tf.keras.models.load_model(model_dir / "image_classifier.h5")
    return loaded_model

def deploy_image_classifier_predictions():
    selected_image = select_image()
    if selected_image is not None:
        image_input = resize_image(selected_image)
        loaded_model = get_trained_image_classifier()
        logits = loaded_model.predict(image_input)
        probas = tf.nn.softmax(logits).numpy()[0]
        class_names = ["Bed", "Chair", "Sofa"]
        st.markdown(
            """
        The model can classify the image as a `Bed`, a `Chair`, or a `Sofa`. 
        Below are the prediction probabilities made by the model.
        """
        )
        df_results = pd.DataFrame(
            {
                "Class Names": class_names,
                "Probabilities": probas
            }
        )
        st.bar_chart(df_results, x="Class Names", y="Probabilities")
        st.markdown(
            f"""
        The model predicts that the image is a **{class_names[np.argmax(probas)]}** with 
        **{max(probas)*100:.2f}%** probability.
        """
        )

