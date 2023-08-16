#!/usr/bin/env python
# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
# Callback to get the time per epoch and write to log
from tensorflow.keras.callbacks import Callback
import mlflow


class EpochTimingCallback(Callback):
    # def __init__(self):
    #     self.batch_times = []
        # self.logs=[]
    def on_epoch_begin(self, epoch, logs=None):
        self.starttime=time()
    def on_epoch_end(self, epoch, logs=None):
        logs['epoch_time'] = (time()-self.starttime)

# Callback to get the time per batch/epoch and write to log
# NB this has a negative impact on performance
class TimingCallback(Callback):
    # def __init__(self):
    #     self.batch_times = []
        # self.logs=[]
    def on_batch_begin(self, batch, logs=None):
        self.batchstart = time()
    def on_batch_end(self, batch, logs=None):
        self.batch_times.append(time() - self.batchstart)
    def on_epoch_begin(self, epoch, logs=None):
        self.starttime=time()
        self.batch_times = []
    def on_epoch_end(self, epoch, logs=None):
        logs['epoch_time'] = (time()-self.starttime)
        mean_batch = np.mean(self.batch_times)
        max_batch = np.max(self.batch_times)
        logs['mean_batch'] = mean_batch
        logs['max_batch'] = max_batch

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
            sw_mse="SQ mean squared error",
            val_hr_sw_loss="Validation HR SW loss",
            val_hr_sw_mae="Validation HR SW mean absolute error",
            val_hr_sw_mse="Validation HR SW mean squared error",
            val_loss="Validation loss",
            val_sw_loss="Validation SW loss",
            val_sw_mae="Validation SW mean absolute error",
            val_sw_mse="Validation SQ mean squared error"
        )
        self.metrics=dict()
    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            display_name = self.key_to_display_name.get(key)
            if display_name is not None:
                self.metrics[display_name] = value
        mlflow.log_metrics(self.metrics, step=epoch)
