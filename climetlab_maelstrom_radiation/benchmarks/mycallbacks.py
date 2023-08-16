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
