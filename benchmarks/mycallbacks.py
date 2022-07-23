# Callback to get the time per epoch and write to log
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
