#!/usr/bin/env python3# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from __future__ import annotations

import climetlab as cml
from climetlab import Dataset
from climetlab.normalize import normalize_args
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
__version__ = "0.1.0"

URL = "https://storage.ecmwf.europeanweather.cloud"

PATTERN = "{url}/MAELSTROM_AP3/TFR/TripCloud{timestep}.{filenum}.tfrecord"

features_size = {
    'sca_inputs' : [17],
    'col_inputs' : [137, 27],
    'hl_inputs' : [138,2],
    'pressure_hl' : [138,1],
    'inter_inputs' : [136,1],
    'sw' : [138,2],
    'lw' : [138,2],
    'hr_sw' : [137,1],
    'hr_lw' : [137,1],
}

feature_description = {}
for k in features_size:
    feature_description[k] = tf.io.FixedLenFeature(features_size[k], tf.float32)


class radiation_tf(Dataset):
    name = None
    home_page = "-"
    licence = "-"
    documentation = "-"
    citation = "-"

    terms_of_use = (
        "By downloading data from this dataset, you agree to the terms and conditions defined at "
        "https://github.com/ecmwf-lab/climetlab_maelstrom_radiation/LICENSE"
        "If you do not agree with such terms, do not download the data. "
    )

    dataset = None

    def __init__(self, timestep=0,filenum=[0,1],
                 input_fields = ['sca_inputs','col_inputs','hl_inputs','inter_inputs','pressure_hl'],
                 output_fields = ['sw','lw','hr_sw','hr_lw'],
                 minimal=False,norm=None):
        self.timestep = timestep
        self.filenum = filenum
        self.input_fields = input_fields
        self.output_fields = output_fields
        self._load()
        if minimal:
            self.sparsefunc = self.sparsen_data
            self.sparse_outputs = Intersection(['sw','lw'],self.output_fields)
        else:
            self.sparsefunc = self.emptyfunction
        if norm is None:
            self.normfunc = self.emptyfunction
        else:
            assert False, "Not yet written norm loader!"

    def emptyfunction(self,x,y):
        return x,y

    def norm_field(self,key,field):
        return (field - self.input_means[key])/self.input_stds[key]
        
    def normalise(self,inputs,outputs):
        for k in inputs:
            inputs[k] = self.norm_field(k,inputs[k])
        return inputs, outputs
        
    def _load(self):
        request = dict(timestep=self.timestep, url=URL, filenum=self.filenum)
        self.source = cml.load_source("url-pattern", PATTERN, request, merger=Merger())
        
    def sparsen_data(self,inputs,outputs):
        for k in self.sparse_outputs:
            outputs[k] = tf.stack([outputs[k][...,0,1],outputs[k][...,-1,0],
                                   outputs[k][...,-1,1]],axis=-1)
        return inputs,outputs

    def _parse_batch(self,record_batch):
        # Create a description of the features
        global feature_description
        # Parse the input `tf.Example` proto using the dictionary above
        example = tf.io.parse_example(record_batch, feature_description)
        inputs = {}
        outputs = {}
        for k in self.input_fields:
            inputs[k] = example[k]
        for k in self.output_fields:
            outputs[k] = example[k]
        return self.sparsefunc(*self.normfunc(inputs,outputs))

    def to_tfrecord(self,
                     batch_size=256,shuffle_size=2048*16,
                     repeat=False):
        # try:
        ds = self.source.to_tfrecord()
        # except:
        #     file_paths = [source._reader.path for source in self.source.sources]
        #     files_ds = tf.data.Dataset.list_files(file_paths)
        #     ignore_order = tf.data.Options()
        #     ignore_order.experimental_deterministic = False
        #     files_ds = files_ds.with_options(ignore_order)
        #     ds = tf.data.TFRecordDataset(files_ds,
        #                                  num_parallel_reads=AUTOTUNE)

        ds = ds.shuffle(shuffle_size)
        # Prepare batches    
        ds = ds.batch(batch_size)
        
        # Parse a batch into a dataset
        ds = ds.map(lambda x: self._parse_batch(x))

        if repeat:
            ds = ds.repeat()

        return ds.prefetch(buffer_size=AUTOTUNE)

class Merger:
    def __init__(self):
        return

    def merge(self,paths):
        files_ds = tf.data.Dataset.list_files(paths)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)
        return tf.data.TFRecordDataset(files_ds,
                                     num_parallel_reads=AUTOTUNE)

def Intersection(lst1, lst2):
    return set(lst1).intersection(lst2)
