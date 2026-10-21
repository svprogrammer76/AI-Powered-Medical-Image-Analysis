"""Small baseline CNN for 28x28 grayscale benchmark images."""
from tensorflow import keras
from tensorflow.keras import layers

from src.data import IMAGE_SHAPE


def build_model(input_shape=IMAGE_SHAPE, compile_model=True):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="pneumonia_cnn",
    )
    if compile_model:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )
    return model
