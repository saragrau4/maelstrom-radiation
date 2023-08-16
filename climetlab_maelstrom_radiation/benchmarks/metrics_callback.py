from tensorflow.keras.callbacks import Callback
import mlflow

class MetricsCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.key_to_display_name= dict(
            hr_sw_loss="HR SW loss",
            hr_sw_mae="HR SW mean absolute error",
            hr_sw_mse="HR SW mean squared error",
            loss="loss",
            lr="Learnig rate",
            sw_loss="SW loss",
            sw_mae="SW mean absolute error",
            sw_mse="SW mean squared error",
            val_hr_sw_loss="Validation HR SW loss",
            val_hr_sw_mae="Validation HR SW mean absolute error",
            val_hr_sw_mse="Validation HR SW mean squared error",
            val_loss="Validation loss",
            val_sw_loss="Validation SW loss",
            val_sw_mae="Validation SW mean absolute error",
            val_sw_mse="Validation SW mean squared error"
        )
        self.metrics=dict()
    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            display_name = self.key_to_display_name.get(key)
            if display_name is not None:
                self.metrics[display_name] = value
        mlflow.log_metrics(self.metrics, step=epoch)