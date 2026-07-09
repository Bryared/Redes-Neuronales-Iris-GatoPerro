# Redes Neuronales — Caso Iris + Caso Cats vs Dogs

Proyecto completo para el curso de Redes Neuronales (UNALM), basado en `Sem 12s1 RN casos.pdf`. Incluye los dos casos de las slides, cada uno implementado en **R y Python**, con análisis detallado de conceptos y arquitecturas, más dos versiones de una app interactiva (**Streamlit** y **Dash**) para probar los modelos entrenados.

**Nota pedagógica**: Todos los archivos `.Rmd` y `.ipynb` incluyen explicaciones paso a paso con un **esquema de colores consistente** (turquesa para conceptos, naranja para objetivos, verde para resultados), inspirado en el documento `EC2_InferenciaEstadistica_Detallado.pdf` del mismo repositorio. Esto facilita la lectura y el aprendizaje de conceptos abstractos (por qué arquitectura, qué significa cada parámetro, cómo interpretar resultados).

```
proyecto/
├── README.md                     # este archivo
├── requirements.txt              # dependencias Python (venv)
├── .venv/                        # entorno virtual Python 3.12
├── data/
│   └── cats_and_dogs_filtered/   # dataset descargado (2000 train / 1000 val)
├── caso1_iris/
│   ├── iris_neuralnet.Rmd        # R (neuralnet): MLP 4→(10,10)→3
│   ├── iris_neuralnet.html       # resultado renderizado
│   ├── iris_keras.ipynb          # Python (Keras): MLP equivalente
│   └── ...outputs/
├── caso2_catsdogs/
│   ├── catsdogs_keras.ipynb      # Python (CNN): entrena y guarda modelo
│   ├── catsdogs_r.Rmd            # R (Keras): carga modelo preentrenado de Python
│   ├── catsdogs_r.html           # resultado renderizado
│   └── ...outputs/
├── app/
│   ├── app_streamlit.py          # versión Streamlit (tabs Iris / Cats vs Dogs)
│   ├── app_dash.py               # versión Dash (callbacks + dcc.Upload)
│   └── models/
│       ├── iris_model.keras      # modelo entrenado Caso 1 Python
│       └── catsdogs_model.keras  # modelo entrenado Caso 2 Python
└── EC2_InferenciaEstadistica_CC3025/  # (proyecto separado)
    ├── EC2_InferenciaEstadistica_Detallado.tex
    └── EC2_InferenciaEstadistica_Detallado.pdf
```

## Instalación rápida

### 1. Clonar/descargar el proyecto y entrar en la carpeta

```bash
cd RedesNeuronales_IrisCatsDogs
```

### 2. Crear entorno virtual Python

Se usa Python 3.12 (TensorFlow no soporta 3.13/3.14 todavía):

```bash
py -3.12 -m venv .venv
./.venv/Scripts/activate     # Windows
# o: source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 3. Instalar paquetes R (opcional, solo si van a usar R)

Abrir R o RStudio y ejecutar:

```r
install.packages("neuralnet")
install.packages("keras")
install.packages("reticulate")
```

## Caso 1 — Clasificación Iris

### R: `iris_neuralnet.Rmd`

Implementación fiel a las slides usando el paquete `neuralnet`:

- **Red**: 4 entradas → capa oculta 1 (10 neuronas) → capa oculta 2 (10 neuronas) → 3 salidas (one-hot)
- **Split**: 70% entrenamiento / 30% validación (semilla 123 para reproducibilidad)
- **Función de error**: entropía cruzada (`err.fct="ce"`)
- **Salida**: matriz de confusión, tasa de error, diagrama de la red

**Cómo ejecutar:**

```bash
# Opción 1: desde RStudio, abrir iris_neuralnet.Rmd y hacer Knit
# Opción 2: desde consola
Rscript -e 'rmarkdown::render("caso1_iris/iris_neuralnet.Rmd")'
```

Genera: `caso1_iris/iris_neuralnet.html` con todos los resultados.

### Python: `iris_keras.ipynb`

Implementación equivalente en Keras/TensorFlow con la **misma arquitectura** que R:

- Mismo dataset (`sklearn.datasets.load_iris`)
- Mismo split 70/30 con semilla 123
- MLP: Input(4) → Dense(10, relu) → Dense(10, relu) → Dense(3, softmax)
- Categorical cross-entropy + Adam optimizer
- Reporta matriz de confusión y accuracy; guarda modelo en `app/models/iris_model.keras`

**Cómo ejecutar:**

```bash
jupyter notebook caso1_iris/iris_keras.ipynb
# Ejecutar todas las celdas (Run All)
```

o de forma automatizada:

```bash
./.venv/Scripts/python -m jupyter nbconvert --to notebook --execute --inplace caso1_iris/iris_keras.ipynb
```

### Comparación R vs Python

Ambas implementaciones usan exactamente la misma arquitectura y los mismos datos con el mismo seed. Las métricas finales (tasa de error, matriz de confusión) deben ser prácticamente idénticas, validando que ambos lenguajes entrenan correctamente el mismo modelo.

## Caso 2 — Clasificación Binaria de Imágenes (Cats vs Dogs)

El dataset original de Google (`mledu-datasets/cats_and_dogs_filtered.zip`) ya no es accesible públicamente. Este proyecto usa en su lugar el dataset completo de Microsoft/Kaggle y prepara automáticamente el mismo subconjunto: **2000 imágenes de entrenamiento + 1000 de validación** (1000 gatos, 1000 perros).

### Python: `catsdogs_keras.ipynb`

CNN entrenada fiel a las slides:

- **Dataset**: descarga automática desde Microsoft/Kaggle, extrae, filtra imágenes corruptas
- **Preprocesamiento**: rescale 1/255, resize 150×150, batch size 20
- **Arquitectura**:
  ```
  Conv2D(16, 3, relu) → MaxPooling2D(2)
  Conv2D(32, 3, relu) → MaxPooling2D(2)
  Conv2D(64, 3, relu) → MaxPooling2D(2)
  Flatten → Dense(512, relu) → Dense(1, sigmoid)  [binary output]
  ```
- **Entrenamiento**: 15 épocas, RMSprop + binary cross-entropy
- **Salida**: curvas accuracy/loss (muestra overfitting típico: train acc ~98%, val acc ~71%), visualización de feature maps de capas intermedias, modelo guardado

**Cómo ejecutar:**

```bash
jupyter notebook caso2_catsdogs/catsdogs_keras.ipynb
# Ejecutar todas las celdas (Run All)
# La primera celda descarga ~800 MB de dataset; tarda un rato la primera vez
```

o:

```bash
./.venv/Scripts/python -m jupyter nbconvert --to notebook --execute --inplace caso2_catsdogs/catsdogs_keras.ipynb
```

Genera:
- Dataset en `data/cats_and_dogs_filtered/`
- Modelo entrenado en `app/models/catsdogs_model.keras` (~76 MB)

### R: `catsdogs_r.Rmd`

Demuestra **interoperabilidad R ↔ Python**: carga el modelo preentrenado de Python (`catsdogs_model.keras`) y lo evalúa en R sobre los mismos datos de validación.

- Carga del modelo: `load_model_tf("../app/models/catsdogs_model.keras")`
- Evaluación sobre 1000 imágenes de validación
- Predicciones individuales sobre muestras de gatos/perros
- **Concepto clave**: el modelo entrenado en Python es agnóstico del lenguaje; R lo reutiliza sin reentrenamiento

**Requisito previo**: ejecutar `catsdogs_keras.ipynb` primero (genera el modelo guardado y el dataset).

**Cómo ejecutar:**

```bash
Rscript -e 'rmarkdown::render("caso2_catsdogs/catsdogs_r.Rmd")'
```

Genera: `caso2_catsdogs/catsdogs_r.html` con evaluación y predicciones en R.

## App Interactiva

Dos implementaciones equivalentes del mismo modelo en una UI:

### Streamlit

```bash
./.venv/Scripts/streamlit run app/app_streamlit.py
```

Abre automáticamente en `http://localhost:8501` con:

- **Tab "Iris"**: sliders para las 4 medidas → botón predecir → clase + barplot de probabilidades
- **Tab "Cats vs Dogs"**: subir imagen → predicción (gato/perro + confianza) + opción de visualizar feature maps de las capas Conv

### Dash

```bash
./.venv/Scripts/python app/app_dash.py
```

Abre en `http://127.0.0.1:8050` con la misma funcionalidad que Streamlit (mismos modelos, mismas predicciones, distintos callbacks/componentes).

**Requisito previo**: haber ejecutado ambos notebooks de Python de cada caso (para generar `iris_model.keras` y `catsdogs_model.keras` en `app/models/`).

## Esquema de Colores y Estilo

Todos los documentos (`.Rmd` y `.ipynb`) siguen el mismo esquema visual para facilitar la lectura:

```
┌─────────────────────────────────────────────────────────────┐
│  🔷 CONCEPTO (Turquesa #20B2AA)                             │
│  Explicación de conceptos: qué es una neurona, por qué      │
│  one-hot encoding, qué significa cada parámetro, etc.       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  🟠 OBJETIVO (Naranja #FF8C00)                              │
│  Qué vamos a hacer en esta sección y por qué.               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  🟢 RESULTADO (Verde #32CD32)                               │
│  Resumen del resultado final, métrica clave, conclusión.    │
└─────────────────────────────────────────────────────────────┘
```

Esto replica el estilo del documento `EC2_InferenciaEstadistica_Detallado.pdf` (un documento LaTeX de estadística con el mismo color scheme), asegurando coherencia visual en todo el proyecto.

## Estructura de Archivos Generados

Después de ejecutar todo:

```
app/models/
├── iris_model.keras          # MLP 4→(10,10)→3, accuracy ~97.8%
└── catsdogs_model.keras      # CNN, train acc ~98.6%, val acc ~71.3%

caso1_iris/
├── iris_neuralnet.Rmd        # código fuente R
├── iris_neuralnet.html       # output renderizado
├── iris_neuralnet_files/     # assets (CSS, plots)
└── iris_keras.ipynb          # notebook Python con outputs

caso2_catsdogs/
├── catsdogs_keras.ipynb      # notebook Python con outputs (76 MB, includes training logs)
├── catsdogs_r.Rmd            # código fuente R
├── catsdogs_r.html           # output renderizado
└── catsdogs_r_files/         # assets

data/
└── cats_and_dogs_filtered/
    ├── train/
    │   ├── cats/ (1000 imágenes)
    │   └── dogs/ (1000 imágenes)
    └── validation/
        ├── cats/ (500 imágenes)
        └── dogs/ (500 imágenes)
```

## Troubleshooting

### TensorFlow no se instala / versiones incompatibles

- **Síntoma**: `pip install tensorflow` falla en Python 3.13/3.14
- **Solución**: usar Python 3.12 (o 3.11, si prefieres). TensorFlow 2.13+ soporta 3.12; versiones más nuevas de Python no son soportadas aún.

### Keras de R no detecta Python venv

- **Síntoma**: error "python not found" o problemas al cargar tensorflow en R
- **Solución**: el `setup` chunk del Rmd fija `RETICULATE_PYTHON` hacia el venv correcto. Si sigue fallando:
  ```r
  Sys.setenv(RETICULATE_PYTHON = normalizePath("./.venv/Scripts/python.exe"))
  library(reticulate)
  py_config()  # debe mostrar el path correcto
  ```

### Imágenes corruptas al descargar dataset

- **Síntoma**: el notebook falla al procesar imágenes durante la extracción del dataset
- **Solución**: el notebook tiene lógica integrada para filtrar imágenes corruptas (`is_valid_image()`), así que si la descarga es limpia, no debe haber problema. Si persiste, revisar los logs de `catsdogs_keras.ipynb`.

### La app Streamlit/Dash no carga los modelos

- **Síntoma**: FileNotFoundError al levantar la app
- **Solución**: asegurarse de haber ejecutado los notebooks de Python (Caso 1 y Caso 2) para generar los modelos `.keras` en `app/models/`.

## Referencias

- **Slides originales**: `Sem 12s1 RN casos.pdf` (del curso)
- **Dataset Iris**: built-in en scikit-learn
- **Dataset Cats vs Dogs**: descargado desde [Kaggle/Microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=54765) (originalmente de Google, ahora restringido)
- **Frameworks**: TensorFlow/Keras 2.13+, R neuralnet, scikit-learn, Streamlit, Dash
- **Ambiente**: Python 3.12 (venv local), R 4.0+

## Contribuciones y Mejoras

Si encontrás bugs o querés agregar más análisis:
1. Creá una rama con tu cambio
2. Testeá que los notebooks/Rmds se ejecuten sin errores
3. Actualizá el README si cambiaste la estructura
4. Push y creá un PR

¡Gracias por usar este proyecto!
