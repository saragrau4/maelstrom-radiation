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

# from typing import List, Optional, OrderedDict, Tuple
# Custom layer for the end of our


@tf.keras.utils.register_keras_serializable()
class TopFlux(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(TopFlux, self).__init__(name=name, **kwargs)
        self.g_cp = tf.constant(9.80665 / 1004 * 24 * 3600)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        fluxes = inputs[0]
        hr = tf.squeeze(inputs[1])
        hlpress = inputs[2]
        # Net surface flux = down - up
        netflux = fluxes[..., 0] - fluxes[..., 1]
        # Pressure difference between the half-levels
        net_press = hlpress[..., 1:, 0] - hlpress[..., :-1, 0]
        # Integrate the heating rate through the atmosphere
        hr_sum = tf.math.reduce_sum(tf.math.multiply(hr, net_press), axis=-1)
        # Stack the outputs
        # TOA net flux, Surface down, #Surface up
        # upwards TOA flux can be deduced as down flux is prescribed
        # either by solar radiation for SW (known) or 0 for LW.
        return tf.stack(
            [netflux + hr_sum / self.g_cp, fluxes[..., 0], fluxes[..., 1]], axis=-1
        )


class rnncolumns(tf.keras.layers.Layer):
    def __init__(self, include_constants=False, name=None, **kwargs):
        super(rnncolumns, self).__init__(name=name, **kwargs)
        colnorms = tf.constant(
            [
                4.61617715e01,
                5.98355832e04,
                2.36960248e03,
                3.01348603e06,
                4.92351671e05,
                4.77463763e00,
                1.16648264e09,
                2.01012275e09,
                1.0,
                1.0,  # <-- Two zero inputs
                1.00000000e00,
                4.12109712e08,
                4.82166968e06,
                3.96867640e06,
                1.97749625e07,
                7.20587302e06,
                7.82937119e06,
                1.66701023e07,
                2.03854471e07,
                2.43620336e08,
                1.37198036e08,
                4.13003711e07,
                2.10871729e09,
                6.47918275e02,
                1.10262260e02,
                3.33333342e04,
                9.93289347e03,
            ],
            dtype=tf.float32,
        )
        if not include_constants:  # remove indices 5 (O2), 8 (hcfc22), 9 (ccl4_vmr)
            colnorms = tf.concat([colnorms[0:5], colnorms[6:8], colnorms[10:]], axis=0)

        self.colnorms = tf.expand_dims(tf.expand_dims(colnorms, axis=0), axis=0)
        self.nlay = tf.constant(137)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def build(self, input_shape):
        pass

    def call(self, inputs):

        fl_inputs, hl_inputs, cos_sza = inputs 
        cos_sza_lay = tf.repeat(
            tf.expand_dims(cos_sza, axis=1), repeats=self.nlay, axis=1
        )
        cos_sza_lay = tf.expand_dims(cos_sza_lay, axis=2)

        fl_inputs2 = tf.concat(
            [fl_inputs[:, :, 0:5], fl_inputs[:, :, 6:8], fl_inputs[:, :, 10:]], axis=2
        )
        fl_inputs2 = tf.math.multiply(fl_inputs2, self.colnorms)

        hl_p = hl_inputs[..., :1]
        temp_hl = hl_inputs[..., 1:]

        # Add pressure to layer inputs
        # Pressure at layers / full levels (137)
        pres_fl = tf.math.multiply(
            tf.constant(0.5), tf.add(hl_p[:, :-1, :], hl_p[:, 1:, :])
        )
        # First normalize
        pres_fl_norm = tf.math.log(pres_fl)
        pres_fl_norm = tf.math.multiply(
            pres_fl_norm, tf.constant(0.086698161)
        )  # scale roughly to 0-1

        temp_fl_norm = tf.multiply(
            tf.constant(0.0031765 * 0.5), tf.add(temp_hl[:, :-1, :], temp_hl[:, 1:, :])
        )

        deltap = tf.math.multiply(
            tf.math.subtract(hl_p[:, 1:, :], hl_p[:, :-1, :]), tf.constant(0.0004561)
        )

        return tf.concat(
            [fl_inputs2, cos_sza_lay, pres_fl_norm, temp_fl_norm, deltap], axis=-1
        )

class rnncolumns_old(tf.keras.layers.Layer):
    def __init__(self, include_constants=False, name=None, **kwargs):
        super(rnncolumns_old, self).__init__(name=name, **kwargs)
        colnorms = tf.constant(
            [
                4.61617715e01,
                5.98355832e04,
                2.36960248e03,
                3.01348603e06,
                4.92351671e05,
                4.77463763e00,
                1.16648264e09,
                2.01012275e09,
                1.0,
                1.0,  # <-- Two zero inputs
                1.00000000e00,
                4.12109712e08,
                4.82166968e06,
                3.96867640e06,
                1.97749625e07,
                7.20587302e06,
                7.82937119e06,
                1.66701023e07,
                2.03854471e07,
                2.43620336e08,
                1.37198036e08,
                4.13003711e07,
                2.10871729e09,
                6.47918275e02,
                1.10262260e02,
                3.33333342e04,
                9.93289347e03,
            ],
            dtype=tf.float32,
        )
        if not include_constants:  # remove indices 5 (O2), 8 (hcfc22), 9 (ccl4_vmr)
            colnorms = tf.concat([colnorms[0:5], colnorms[6:8], colnorms[10:]], axis=0)

        self.colnorms = tf.expand_dims(tf.expand_dims(colnorms, axis=0), axis=0)
        self.nlay = tf.constant(137)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def build(self, input_shape):
        pass

    def call(self, fl_inputs, hl_inputs, cos_sza):

        cos_sza_lay = tf.repeat(
            tf.expand_dims(cos_sza, axis=1), repeats=self.nlay, axis=1
        )
        cos_sza_lay = tf.expand_dims(cos_sza_lay, axis=2)

        fl_inputs2 = tf.concat(
            [fl_inputs[:, :, 0:5], fl_inputs[:, :, 6:8], fl_inputs[:, :, 10:]], axis=2
        )
        fl_inputs2 = tf.math.multiply(fl_inputs2, self.colnorms)

        hl_p = hl_inputs[..., :1]
        temp_hl = hl_inputs[..., 1:]

        # Add pressure to layer inputs
        # Pressure at layers / full levels (137)
        pres_fl = tf.math.multiply(
            tf.constant(0.5), tf.add(hl_p[:, :-1, :], hl_p[:, 1:, :])
        )
        # First normalize
        pres_fl_norm = tf.math.log(pres_fl)
        pres_fl_norm = tf.math.multiply(
            pres_fl_norm, tf.constant(0.086698161)
        )  # scale roughly to 0-1

        temp_fl_norm = tf.multiply(
            tf.constant(0.0031765 * 0.5), tf.add(temp_hl[:, :-1, :], temp_hl[:, 1:, :])
        )

        deltap = tf.math.multiply(
            tf.math.subtract(hl_p[:, 1:, :], hl_p[:, :-1, :]), tf.constant(0.0004561)
        )

        return tf.concat(
            [fl_inputs2, cos_sza_lay, pres_fl_norm, temp_fl_norm, deltap], axis=-1
        )



@tf.keras.utils.register_keras_serializable()
class HRLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        name=None,
        hr_units="K d-1",
        **kwargs,
    ):
        super(HRLayer, self).__init__(name=name, **kwargs)
        time_scale = {"K s-1": 1, "K d-1": 3600 * 24}[hr_units]
        self.g_cp = tf.constant(9.80665 / 1004 * time_scale, dtype=tf.float32)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def build(self, input_shape):
        pass

    def call(self, inputs):
        fluxes = inputs[0]
        hlpress = inputs[1]
        netflux = fluxes[..., 0] - fluxes[..., 1]
        flux_diff = netflux[..., 1:] - netflux[..., :-1]
        net_press = hlpress[..., 1:, 0] - hlpress[..., :-1, 0]
        return -self.g_cp * tf.math.divide(flux_diff, net_press)

@tf.keras.utils.register_keras_serializable()
class AddHeight(tf.keras.layers.Layer):
    def __init__(self, name=None,shape=(1,138,1),trainable=True,dtype=None):
        super(AddHeight, self).__init__(name=name,trainable=trainable,dtype=dtype)
        self.shape = shape

    def build(self, input_shape):
        self.b = self.add_weight(shape=self.shape,
                                 initializer='random_normal',
                                 trainable=True)
    def get_config(self):
        cfg = super().get_config()
        return cfg

    def call(self, inputs):
        return inputs + self.b
