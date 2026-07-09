import os

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from tensorflow import keras

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
IRIS_MODEL_PATH = os.path.join(MODELS_DIR, "iris_model.keras")
CATSDOGS_MODEL_PATH = os.path.join(MODELS_DIR, "catsdogs_model.keras")

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]

st.set_page_config(page_title="Redes Neuronales - Iris & Cats vs Dogs", layout="wide")
st.title("Redes Neuronales: Iris + Cats vs Dogs")

tab_iris, tab_catsdogs = st.tabs(["🌸 Iris", "🐱🐶 Cats vs Dogs"])


@st.cache_resource
def load_iris_model():
    return keras.models.load_model(IRIS_MODEL_PATH)


@st.cache_resource
def load_catsdogs_model():
    return keras.models.load_model(CATSDOGS_MODEL_PATH)


with tab_iris:
    st.header("Clasificación de flores Iris")
    st.caption("Red neuronal (MLP, 2 capas ocultas de 10 neuronas) entrenada sobre las 150 observaciones del dataset Iris.")

    if not os.path.isfile(IRIS_MODEL_PATH):
        st.warning("No se encontró el modelo entrenado. Ejecuta primero caso1_iris/iris_keras.ipynb.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
            sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
        with col2:
            petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.3, 0.1)
            petal_width = st.slider("Petal width (cm)", 0.1, 2.6, 1.3, 0.1)

        if st.button("Predecir especie"):
            model = load_iris_model()
            x = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            probs = model.predict(x, verbose=0)[0]
            pred_idx = int(np.argmax(probs))

            st.success(f"Especie predicha: **{IRIS_CLASSES[pred_idx]}**")

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(IRIS_CLASSES, probs, color=["#4C72B0", "#DD8452", "#55A868"])
            ax.set_ylabel("Probabilidad")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

with tab_catsdogs:
    st.header("Clasificación de imágenes: Gato vs Perro")
    st.caption("CNN (3 capas convolucionales + densa) entrenada sobre el dataset Cats vs Dogs.")

    if not os.path.isfile(CATSDOGS_MODEL_PATH):
        st.warning("No se encontró el modelo entrenado. Ejecuta primero caso2_catsdogs/catsdogs_keras.ipynb.")
    else:
        uploaded_file = st.file_uploader("Sube una imagen de un gato o un perro", type=["jpg", "jpeg", "png"])

        show_feature_maps = st.checkbox("Mostrar representación intermedia (feature maps)", value=False)

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Imagen subida", width=300)

            model = load_catsdogs_model()
            img_resized = image.resize((150, 150))
            x = np.array(img_resized) / 255.0
            x = np.expand_dims(x, axis=0)

            prob_dog = float(model.predict(x, verbose=0)[0, 0])
            label = "Perro 🐶" if prob_dog >= 0.5 else "Gato 🐱"
            confidence = prob_dog if prob_dog >= 0.5 else 1 - prob_dog

            st.success(f"Predicción: **{label}** (confianza: {confidence:.1%})")

            fig, ax = plt.subplots(figsize=(5, 1.2))
            ax.barh(["Perro"], [prob_dog], color="#DD8452")
            ax.set_xlim(0, 1)
            ax.set_xlabel("P(perro)")
            st.pyplot(fig)

            if show_feature_maps:
                st.subheader("Representación intermedia de las capas convolucionales")
                conv_layers = [l for l in model.layers if "conv2d" in l.name or "max_pooling2d" in l.name]
                successive_outputs = [layer.output for layer in conv_layers]
                visualization_model = keras.Model(model.input, successive_outputs)
                feature_maps = visualization_model.predict(x, verbose=0)

                for layer, fmap in zip(conv_layers, feature_maps):
                    n_features = fmap.shape[-1]
                    size = fmap.shape[1]
                    n_show = min(n_features, 16)
                    display_grid = np.zeros((size, size * n_show))
                    for i in range(n_show):
                        f = fmap[0, :, :, i]
                        f -= f.mean()
                        f /= (f.std() + 1e-5)
                        f *= 64
                        f += 128
                        f = np.clip(f, 0, 255).astype("uint8")
                        display_grid[:, i * size:(i + 1) * size] = f
                    scale = 20.0 / n_show
                    fig2, ax2 = plt.subplots(figsize=(scale * n_show, scale))
                    ax2.set_title(layer.name)
                    ax2.axis("off")
                    ax2.imshow(display_grid, aspect="auto", cmap="viridis")
                    st.pyplot(fig2)
