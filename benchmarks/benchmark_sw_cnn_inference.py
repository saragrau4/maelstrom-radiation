from time import time

import os
# To hide GPU uncomment
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.layers import Input,Conv1D,Add,RepeatVector,ZeroPadding1D,Reshape
from tensorflow.keras.layers import Cropping1D,Concatenate,Multiply,Flatten,Dense
from tensorflow import nn
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
import climetlab as cml

import numpy as np
norms = np.load('inp_max_norm.npy',allow_pickle=True)
norms = norms[()]
for key in norms:
    norms[key][norms[key]==0] = 1


def load_data(batch_size = 256, sample_data = False,
              synthetic_data = False,
          ):

    kwargs = {'hr_units':'K d-1',
              'norm':False,
              'minimal_outputs':True,
              'topnetflux':True,
              'dataset':'tripleclouds',
              'output_fields':['sw','hr_sw']}
    if sample_data:
        train_ts = [0]
        train_fn = [0]
        val_ts = 2019013100
        val_fn = [0]
        train_num = 265 * 256 // batch_size
        val_num = 265 * 256 // batch_size
    else:
        train_ts = list(range(0,3501,1000)) # 500))
        train_fn = list(range(0,51,5))
        val_ts = 2019013100
        val_fn = [0, 25, 50] # 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        val_num = 795 * 256 // batch_size
        train_num = 11660 * 256 // batch_size
        
    ds_cml = cml.load_dataset('maelstrom-radiation-tf',
                              timestep = train_ts,
                              filenum = train_fn,
                              **kwargs)
    
    train = ds_cml.to_tfdataset(batch_size=batch_size,shuffle=True)
    
    if synthetic_data:
        print("Creating synthetic data by repeating single batch, useful for pipeline testing.")
        train = train.take(1).cache().repeat(train_num)

    return train

# Custom layer for the end of our NN
@tf.keras.utils.register_keras_serializable()
class TopFlux(tf.keras.layers.Layer):
    def __init__(self,name=None,**kwargs):
        super(TopFlux, self).__init__(name=name,**kwargs)
        self.g_cp = tf.constant(9.80665 / 1004 * 24 * 3600)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        fluxes = inputs[0]
        hr = tf.squeeze(inputs[1])
        hlpress = inputs[2]
        #Net surface flux = down - up
        netflux = fluxes[...,0] - fluxes[...,1]
        #Pressure difference between the half-levels
        net_press = hlpress[...,1:,0]-hlpress[...,:-1,0]
        #Integrate the heating rate through the atmosphere
        hr_sum = tf.math.reduce_sum(tf.math.multiply(hr,net_press),axis=-1)
        #Stack the outputs
        # TOA net flux, Surface down, #Surface up
        # upwards TOA flux can be deduced as down flux is prescribed
        # either by solar radiation for SW (known) or 0 for LW.
        return tf.stack([netflux + hr_sum / self.g_cp,fluxes[...,0],fluxes[...,1]],axis=-1)


#Here we construct a model.
    
def load_model(
        model_name
              ):
    model = tf.keras.models.load_model(model_name,
                                       custom_objects = {'TopFlux':TopFlux})
    return model


def main(model_name = "trained_model.h5",
         batch_size = 256, sample_data = False,
         synthetic_data = False, tensorboard = False, 
):
    print("Getting training/validation data")
    total_start = time()
    train = load_data(batch_size = batch_size,
                          sample_data = sample_data,
                          synthetic_data = synthetic_data,
    )
    load_time = time() - total_start
    model = load_model(model_name    )

    infer_start = time()
    # for inp,out in train:
    #     pred = model.predict(inp)
    model.evaluate(train,verbose=2)
    infer_time = time() - infer_start

    total_time = time() - total_start

    print("---------------------------------------------------")
    print("Timing statistics")
    print("---------------------------------------------------")
    print("Total time: ",total_time)
    print("Load time: ",load_time)
    print("Infer time: ",infer_time)
    print("---------------------------------------------------")
    
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run benchmark')
    parser.add_argument('--sample_data',help="Use small dataset for bugtesting", action='store_const',
                        const = True, default = False)
    parser.add_argument('--synthetic_data',help="Use synthetic dataset for pipeline testing", action='store_const',
                        const = True, default = False)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--model', type=str, default="trained_model.h5")
    args = parser.parse_args()
    main(model_name = args.model,
        batch_size = args.batch,
        sample_data = args.sample_data,
        synthetic_data = args.synthetic_data,
    )
