import base64
import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from tensorflow import keras

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
IRIS_MODEL_PATH = os.path.join(MODELS_DIR, "iris_model.keras")
CATSDOGS_MODEL_PATH = os.path.join(MODELS_DIR, "catsdogs_model.keras")

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]

iris_model = keras.models.load_model(IRIS_MODEL_PATH) if os.path.isfile(IRIS_MODEL_PATH) else None
catsdogs_model = keras.models.load_model(CATSDOGS_MODEL_PATH) if os.path.isfile(CATSDOGS_MODEL_PATH) else None

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Redes Neuronales - Iris & Cats vs Dogs"
server = app.server  # expuesto para gunicorn en producción (Render, etc.)


def fig_to_src(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def iris_slider(id_, label, min_, max_, value):
    return html.Div(
        [
            html.Label(label),
            dcc.Slider(
                id=id_, min=min_, max=max_, value=value, step=0.1,
                marks=None, tooltip={"placement": "bottom", "always_visible": True},
            ),
        ],
        style={"marginBottom": "1.5rem"},
    )


iris_tab = dbc.Card(
    dbc.CardBody(
        [
            html.P(
                "Red neuronal (MLP, 2 capas ocultas de 10 neuronas) entrenada sobre las "
                "150 observaciones del dataset Iris.",
                className="text-muted",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            iris_slider("sepal-length", "Sepal length (cm)", 4.0, 8.0, 5.8),
                            iris_slider("sepal-width", "Sepal width (cm)", 2.0, 4.5, 3.0),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            iris_slider("petal-length", "Petal length (cm)", 1.0, 7.0, 4.3),
                            iris_slider("petal-width", "Petal width (cm)", 0.1, 2.6, 1.3),
                        ],
                        width=6,
                    ),
                ]
            ),
            dbc.Button("Predecir especie", id="iris-predict-btn", color="primary"),
            html.Div(id="iris-result", style={"marginTop": "1.5rem"}),
        ]
    )
)

catsdogs_tab = dbc.Card(
    dbc.CardBody(
        [
            html.P(
                "CNN (3 capas convolucionales + densa) entrenada sobre el dataset Cats vs Dogs.",
                className="text-muted",
            ),
            dcc.Upload(
                id="upload-image",
                children=html.Div(["Arrastra una imagen o ", html.A("selecciónala")]),
                style={
                    "width": "100%", "height": "80px", "lineHeight": "80px",
                    "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "8px",
                    "textAlign": "center", "marginBottom": "1rem",
                },
                accept="image/*",
                multiple=False,
            ),
            dbc.Checkbox(
                id="show-feature-maps",
                label="Mostrar representación intermedia (feature maps)",
                value=False,
                style={"marginBottom": "1rem"},
            ),
            html.Div(id="catsdogs-result"),
        ]
    )
)

app.layout = dbc.Container(
    [
        html.H2("Redes Neuronales: Iris + Cats vs Dogs", style={"marginTop": "1.5rem"}),
        dbc.Tabs(
            [
                dbc.Tab(iris_tab, label="🌸 Iris"),
                dbc.Tab(catsdogs_tab, label="🐱🐶 Cats vs Dogs"),
            ]
        ),
    ],
    fluid=False,
    style={"maxWidth": "900px", "paddingBottom": "3rem"},
)


@app.callback(
    Output("iris-result", "children"),
    Input("iris-predict-btn", "n_clicks"),
    State("sepal-length", "value"),
    State("sepal-width", "value"),
    State("petal-length", "value"),
    State("petal-width", "value"),
    prevent_initial_call=True,
)
def predict_iris(n_clicks, sl, sw, pl, pw):
    if iris_model is None:
        return dbc.Alert("No se encontró el modelo entrenado. Ejecuta antes iris_keras.ipynb.", color="warning")

    x = np.array([[sl, sw, pl, pw]])
    probs = iris_model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(IRIS_CLASSES, probs, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("Probabilidad")
    ax.set_ylim(0, 1)
    src = fig_to_src(fig)

    return html.Div(
        [
            dbc.Alert(f"Especie predicha: {IRIS_CLASSES[pred_idx]}", color="success"),
            html.Img(src=src, style={"maxWidth": "100%"}),
        ]
    )


@app.callback(
    Output("catsdogs-result", "children"),
    Input("upload-image", "contents"),
    Input("show-feature-maps", "value"),
    prevent_initial_call=True,
)
def predict_catsdogs(contents, show_feature_maps):
    if catsdogs_model is None:
        return dbc.Alert("No se encontró el modelo entrenado. Ejecuta antes catsdogs_keras.ipynb.", color="warning")
    if contents is None:
        return html.Div()

    header, encoded = contents.split(",", 1)
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

    img_resized = image.resize((150, 150))
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    prob_dog = float(catsdogs_model.predict(x, verbose=0)[0, 0])
    label = "Perro 🐶" if prob_dog >= 0.5 else "Gato 🐱"
    confidence = prob_dog if prob_dog >= 0.5 else 1 - prob_dog

    children = [
        html.Img(src=contents, style={"maxWidth": "300px", "display": "block", "marginBottom": "1rem"}),
        dbc.Alert(f"Predicción: {label} (confianza: {confidence:.1%})", color="success"),
    ]

    if show_feature_maps:
        conv_layers = [l for l in catsdogs_model.layers if "conv2d" in l.name or "max_pooling2d" in l.name]
        successive_outputs = [layer.output for layer in conv_layers]
        visualization_model = keras.Model(catsdogs_model.input, successive_outputs)
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
            children.append(html.Img(src=fig_to_src(fig2), style={"maxWidth": "100%", "marginBottom": "0.5rem"}))

    return html.Div(children)


if __name__ == "__main__":
    # host="0.0.0.0" y el puerto desde $PORT son necesarios para que servicios como
    # Render/Railway puedan enrutar tráfico externo hacia el contenedor.
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
