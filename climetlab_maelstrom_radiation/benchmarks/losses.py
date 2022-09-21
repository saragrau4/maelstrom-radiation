#!/usr/bin/env python
# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()
huber = tf.keras.losses.Huber()


def top_scaledflux_mse(y_true, y_pred):
    sca = y_true[..., :1, :1]
    y_true_tmp = y_true / sca
    y_pred_tmp = y_pred / sca
    return mse(y_true_tmp, y_pred_tmp)


def top_scaledflux_mae(y_true, y_pred):
    sca = y_true[..., :1, :1]
    y_true_tmp = y_true / sca
    y_pred_tmp = y_pred / sca
    return mae(y_true_tmp, y_pred_tmp)


def top_scaledflux_huber(y_true, y_pred):
    sca = y_true[..., :1, :1]
    y_true_tmp = y_true / sca
    y_pred_tmp = y_pred / sca
    return huber(y_true_tmp, y_pred_tmp)
