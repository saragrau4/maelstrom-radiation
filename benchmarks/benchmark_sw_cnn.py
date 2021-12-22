import os
# To hide GPU uncomment
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input,Conv1D,Add,RepeatVector,ZeroPadding1D,Reshape
from tensorflow.keras.layers import Cropping1D,Concatenate,Multiply,Flatten,Dense
from tensorflow import nn
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from time import time

# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
import climetlab as cml

import numpy as np
norms = np.load('inp_max_norm.npy',allow_pickle=True)
norms = norms[()]
for key in norms:
    norms[key][norms[key]==0] = 1
norms

class EpochTimingCallback(Callback):
    # def __init__(self):
    #     self.batch_times = []
        # self.logs=[]
    def on_epoch_begin(self, epoch, logs=None):
        self.starttime=time()
    def on_epoch_end(self, epoch, logs=None):
        logs['epoch_time'] = (time()-self.starttime)

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

def load_data(batch_size = 256, sample_data = False,
              synthetic_data = False
          ):

    if synthetic_data:
        from synthdata import get_synth_dataset
        print("Creating synthetic data for pipeline testing")
        if sample_data:
            train = get_synth_dataset(batch_size, dataset_size = 256 * 265)
            val = get_synth_dataset(batch_size, dataset_size = 256 * 265)
        else:
            train = get_synth_dataset(batch_size, dataset_size = 256 * 11660)
            val = get_synth_dataset(batch_size, dataset_size = 256 * 795)

    else:
            
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
        else:
            train_ts = list(range(0,3501,1000)) # 500))
            train_fn = list(range(0,51,5))
            val_ts = 2019013100
            val_fn = [0, 25, 50] # 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            
        ds_cml = cml.load_dataset('maelstrom-radiation-tf',
                                  timestep = train_ts,
                                  filenum = train_fn,
                                  **kwargs)
        
        train = ds_cml.to_tfdataset(batch_size=batch_size,shuffle=True)

        ds_cml_val = cml.load_dataset('maelstrom-radiation-tf',
                                      timestep = val_ts,
                                      filenum = val_fn,
                                      **kwargs)

        val = ds_cml_val.to_tfdataset(batch_size=batch_size,shuffle=False)

    return train,val

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
    
def buildmodel(
    input_shape,
    output_shape,
    kernel_width = 5,
    conv_filters = 64,
    dilation_rates =  [1,2,4,8,16],
    conv_layers = 6,
    flux_layers = 3,
    flux_filters = 8,
    flux_width = 32,
              ):
    inputs = {}

    #Inputs are a bit messy, in the next few lines we create the input shapes,    
    for k in input_shape.keys():
        inputs[k] = Input(input_shape[k].shape[1:],name=k)
        
    # Calculate the incoming solar radiation before it is normalised.
    in_solar = inputs['sca_inputs'][...,1:2]*inputs['sca_inputs'][...,-1:]
    #Pressure needs to be un-normalised.
    hl_p = inputs['pressure_hl']
    
    #Normalise them by the columnar-maxes
    normed = {}
    for i,k in enumerate(input_shape.keys()):
        normed[k]= inputs[k]/tf.constant(norms[k])
    
    # and repeat or reshape them so they all have 138 vertical layers
    rep_sca = RepeatVector(138)(normed['sca_inputs'])
    col_inp = ZeroPadding1D(padding=(1,0))(normed['col_inputs'])
    inter_inp = ZeroPadding1D(padding=(1,1))(normed['inter_inputs'])
    all_col = Concatenate(axis=-1)([rep_sca,col_inp,normed['hl_inputs'],inter_inp])

    #Use dilation to allow information to propagate faster through the vertical.
    for drate in dilation_rates:
        all_col = Conv1D(filters=conv_filters,kernel_size = kernel_width,
                         dilation_rate = drate,padding='same',
                         data_format='channels_last',
                         activation = nn.swish)(all_col)
    #Regular conv layers
    for i in range(conv_layers):
        all_col = Conv1D(conv_filters,
                         kernel_size=kernel_width,
                         strides=1,padding='same',
                         data_format='channels_last',
                        )(all_col)

    #Predict single output, the heating rate.
    sw = Conv1D(filters=1,padding='same',kernel_size=kernel_width,
                data_format='channels_last',
                activation='linear')(all_col)
    
    #Crop the bottom value to make output correct size.
    #Perhaps you can think of a better solution?
    sw_hr = Cropping1D((0,1),name='hr_sw')(sw)
    
    #Reduce the number of features
    flux_col = Conv1D(filters=flux_filters,padding='same',kernel_size=kernel_width,
                      activation=nn.swish)(all_col)
    flux_col = Flatten()(flux_col)
    swf = Concatenate(axis=-1)([flux_col,normed['sca_inputs']])
    #Add a few dense layers, plus the input scalars
    for i in range(flux_layers):
        swf = Dense(flux_width,activation=nn.swish)(swf)
    swf = Dense(2,activation = 'sigmoid')(swf)
    swf = Multiply()([swf,in_solar])
    swf = TopFlux(name='sw')([swf,sw_hr,hl_p])

    
    #Dictionary of outputs
    output = {'hr_sw':sw_hr,
             'sw':swf}

    model = Model(inputs,output)
    model.compile(loss={'hr_sw':'mse',
                        'sw':'mse'},
                  loss_weights = {'hr_sw':10**(3),
                        'sw':1},
                  optimizer=Adam(10**(-5)))
    return model


def main(batch_size = 256, epochs = 5, sample_data = False,
         synthetic_data = False,
):
    print("Getting training/validation data")
    train,val = load_data(batch_size = batch_size,
                          sample_data = sample_data,
                          synthetic_data = synthetic_data,
    )
    model = buildmodel(train.element_spec[0], 
                       train.element_spec[1])

    callbacks = [ EpochTimingCallback(), # TimingCallback(),
                  tf.keras.callbacks.CSVLogger('training.log'),
                  # tf.keras.callbacks.TensorBoard(log_dir='/data/mchantry/tensorboard',
                  #                               update_freq = 'batch',
                  #                               write_steps_per_second = True,
                  #                               histogram_freq = 0),
    ]
    hist = model.fit(train, 
                     validation_data=val, 
                     epochs = epochs,
                     verbose = 2,
                     callbacks = callbacks,
                 )
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run benchmark')
    parser.add_argument('--sample_data',help="Use small dataset for bugtesting", action='store_const',
                        const = True, default = False)
    parser.add_argument('--synthetic_data',help="Use synthetic dataset for pipeline testing", action='store_const',
                        const = True, default = False)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    main(batch_size = args.batch,
         epochs = args.epochs,
         sample_data = args.sample_data,
         synthetic_data = args.synthetic_data
)
