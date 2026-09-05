import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

st.set_page_config(
    page_title="Skin Lesion Classification",
    page_icon="🩺",
    layout="centered"
)

IMG_SIZE = (224, 224)
MODEL_PATH = "best_transformer_model.keras"


class Patches(layers.Layer):
    def __init__(self, patch_size=16, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=projection_dim,
        )

    def call(self, patch):
        positions = tf.range(
            start=0,
            limit=self.num_patches,
            delta=1,
        )
        encoded = self.projection(patch)
        return encoded + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Put best_transformer_model.keras in the same folder as app.py."
        )

    return keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "Patches": Patches,
            "PatchEncoder": PatchEncoder,
        },
        compile=False,
    )


def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.array(image).astype("float32") / 255.0
    array = np.expand_dims(array, axis=0)
    return image, array


st.title("🩺 Skin Lesion Classification")
st.write("### Melanoma vs. Benign")

st.warning(
    "Educational/research project only. "
    "This application is not a medical diagnosis and should not replace a healthcare professional."
)

st.write("Upload a skin lesion image to get the model prediction.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        display_image = Image.open(uploaded_file).convert("RGB")
        st.image(display_image, caption="Uploaded image", use_container_width=True)

        if st.button("Predict"):
            with st.spinner("Loading model and analyzing image..."):
                model = load_model()

                # Re-open the uploaded file for preprocessing.
                uploaded_file.seek(0)
                _, image_array = preprocess_image(uploaded_file)

                probability = float(
                    model.predict(image_array, verbose=0)[0][0]
                )

                if probability >= 0.5:
                    prediction = "Melanoma"
                    confidence = probability
                else:
                    prediction = "Benign"
                    confidence = 1.0 - probability

            st.success(f"Prediction: {prediction}")
            st.metric("Prediction confidence", f"{confidence * 100:.2f}%")

            st.write("#### Model probabilities")
            st.write(f"Melanoma: {probability * 100:.2f}%")
            st.write(f"Benign: {(1.0 - probability) * 100:.2f}%")

    except Exception as e:
        st.error("The application could not run.")
        st.code(str(e))
        st.info(
            "Make sure app.py and best_transformer_model.keras are in the same GitHub repository."
        )
