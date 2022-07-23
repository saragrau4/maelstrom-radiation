import numpy as np
from tensorflow.keras.callbacks import Callback
from time import time


# Callback to get the time per epoch and write to log
class EpochTimingCallback(Callback):
    # def __init__(self):
    #     self.batch_times = []
    # self.logs=[]
    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = time()

    def on_epoch_end(self, epoch, logs=None):
        logs["epoch_time"] = time() - self.starttime


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
        self.starttime = time()
        self.batch_times = []

    def on_epoch_end(self, epoch, logs=None):
        logs["epoch_time"] = time() - self.starttime
        mean_batch = np.mean(self.batch_times)
        max_batch = np.max(self.batch_times)
        logs["mean_batch"] = mean_batch
        logs["max_batch"] = max_batch


def printstats(logfile, total_time, train_time, load_time, save_time, batch_size):

    print("---------------------------------------------------")
    print("Timing statistics")
    print("---------------------------------------------------")
    print("Total time: ", total_time)
    print("Load time: ", load_time)
    print("Train time: ", train_time)
    print("Save time: ", save_time)
    print("---------------------------------------------------")
    data_size = 2984960

    rundata = np.loadtxt(logfile, skiprows=1, delimiter=",")
    epochs = rundata.shape[0]

    av_train = train_time / epochs
    first_ep = rundata[0, 1]
    min_ep = rundata[:, 1].min()
    max_ep = rundata[:, 1].max()
    n_iter = np.ceil(data_size / batch_size) * epochs
    av_iter = train_time / n_iter
    final_loss = rundata[-1, 3]
    final_val = rundata[-1, 6]

    print(
        "load_time, total_time, train_time,"
        + "av_train, first_ep, min_ep, max_ep, "
        + "av_iter, final_loss, final_val, save_time"
    )
    prv = (
        load_time,
        total_time,
        train_time,
        +av_train,
        first_ep,
        min_ep,
        max_ep,
        +av_iter,
        final_loss,
        final_val,
        save_time,
    )
    fmt = ", ".join(["%.6f" for i in range(len(prv))])
    print(fmt % prv)
    return
