"""
Script standalone para entrenar el modelo Cats vs Dogs (V3: Data Augmentation + Dropout +
EarlyStopping/ModelCheckpoint) SIN pasar por Jupyter/nbconvert, así no hay timeout de celda.

Correr desde la carpeta RedesNeuronales_IrisCatsDogs:

    ./.venv/Scripts/python -u caso2_catsdogs/train_standalone.py

El flag -u (unbuffered) hace que veas cada línea de progreso en el instante en que se
imprime, en vez de que Python la acumule en un buffer.

Para dejarlo corriendo en segundo plano y poder cerrar la terminal:

    ./.venv/Scripts/python -u caso2_catsdogs/train_standalone.py > training_log.txt 2>&1 &

Y para ver el progreso en vivo mientras corre en background:

    tail -f training_log.txt          # en Git Bash
    Get-Content training_log.txt -Wait -Tail 20   # en PowerShell
"""
import os

os.makedirs("../app/models", exist_ok=True)

DATA_DIR = "../data"
BASE_DIR = os.path.join(DATA_DIR, "cats_and_dogs_filtered")
train_dir = os.path.join(BASE_DIR, "train")
validation_dir = os.path.join(BASE_DIR, "validation")

assert os.path.isdir(train_dir), (
    "No existe el dataset preparado. Corré primero la primera celda de "
    "catsdogs_keras.ipynb para descargar/preparar data/cats_and_dogs_filtered/"
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=20, class_mode="binary",
)
validation_generator = val_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=20, class_mode="binary",
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input

model = Sequential(
    [
        Input(shape=(150, 150, 3)),
        Conv2D(16, 3, activation="relu"),
        MaxPooling2D(2),
        Dropout(0.2),
        Conv2D(32, 3, activation="relu"),
        MaxPooling2D(2),
        Dropout(0.2),
        Conv2D(64, 3, activation="relu"),
        MaxPooling2D(2),
        Dropout(0.3),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ]
)
model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(
    loss="binary_crossentropy",
    optimizer=RMSprop(learning_rate=0.001),
    metrics=["acc"],
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    ModelCheckpoint(
        "../app/models/catsdogs_model_best.keras",
        monitor="val_acc",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_acc",
        patience=8,
        restore_best_weights=True,
        mode="max",
        verbose=1,
    ),
]

print("\n" + "=" * 70)
print("INICIANDO ENTRENAMIENTO — puede tardar 30-60+ min según tu CPU")
print("Se detendrá automáticamente si no mejora en 8 épocas seguidas (EarlyStopping)")
print("=" * 70 + "\n")

history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=60,  # tope alto; EarlyStopping decide cuándo parar de verdad
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=callbacks,
    verbose=2,
)

model.save("../app/models/catsdogs_model.keras")

print("\n" + "=" * 70)
print("ENTRENAMIENTO TERMINADO")
print("=" * 70)
print(f"Épocas corridas: {len(history.history['acc'])}")
print(f"Train acc (última época): {history.history['acc'][-1]:.4f}")
print(f"Val acc (mejor época): {max(history.history['val_acc']):.4f}")
print(f"Val acc (última época): {history.history['val_acc'][-1]:.4f}")
print("\nModelo final guardado en app/models/catsdogs_model.keras")
print("(equivalente a app/models/catsdogs_model_best.keras, que ModelCheckpoint")
print(" fue actualizando automáticamente en cada mejora)")
