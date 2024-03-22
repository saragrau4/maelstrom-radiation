from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import mlflow
import numpy as np
from climetlab_maelstrom_radiation.benchmarks.utils import EpochTimingCallback
from climetlab_maelstrom_radiation.benchmarks.metrics_callback import MetricsCallback

X_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 1)

X_val = np.random.rand(20, 10)
y_val = np.random.rand(20, 1)

model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(1)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae', 'mse'])


with mlflow.start_run():
    model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), callbacks=[MetricsCallback(), EpochTimingCallback()])
