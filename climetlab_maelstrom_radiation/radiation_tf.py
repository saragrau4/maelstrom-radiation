#!/usr/bin/env python3 
# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from __future__ import annotations

import climetlab as cml

# from climetlab.normalize import normalize_args
import tensorflow as tf
import xarray as xr
from climetlab import Dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE
__version__ = "0.1.0"

HEADURL = "https://storage.ecmwf.europeanweather.cloud"
URL = f"{HEADURL}/MAELSTROM_AP3/TFR"

NORM_PATTERN = "{url}/MAELSTROM_AP3/{input_fields}.nc"
PATTERN = "{url}/TripCloud{timestep}.{filenum}.tfrecord"

features_size = {
    "sca_inputs": [17],
    "col_inputs": [137, 27],
    "hl_inputs": [138, 2],
    "pressure_hl": [138, 1],
    "inter_inputs": [136, 1],
    "sw": [138, 2],
    "lw": [138, 2],
    "hr_sw": [137, 1],
    "hr_lw": [137, 1],
}

timestep_subset = {
    "tier-1": 0,
    "2020": list(range(0, 3501, 125)),
    "2019013100": 2019013100,
    "2019053100": 2019053100,
    "2019082900": 2019082900,
    "2019102800": 2019102800,
}
filenum_subset = {
    "tier-1": 0,
    "2020": list(range(52)),
    "2019013100": list(range(116)),
    "2019053100": list(range(116)),
    "2019082900": list(range(116)),
    "2019102800": list(range(116)),
}
_num_norm = 0

feature_description = {}
for k in features_size:
    feature_description[k] = tf.io.FixedLenFeature(features_size[k], tf.float32)


class radiation_tf(Dataset):
    name = "radiation_tf"
    home_page = "https://git.ecmwf.int/projects/MLFET/repos/maelstrom-radiation"
    licence = "CC BY 4.0, see https://apps.ecmwf.int/datasets/licences/general/ "
    documentation = (
        "Minimal call:\n"
        "cml.load_dataset('maelstrom-radiation-tf') \n"
        "Optional arguments:\n"
        "Specify subset, subset = 'tier-1' (NB different subset to raw radiation)\n"
        "Or specify timestep/filenum\n"
        "Valid values found in .valid_timstep etc\n"
        "To remove items from input/outputs change input_fields/output_fields\n"
        "e.g. ouput_fields = ['sw','hr_sw'] to build a model only predicting the sw heating\n"
        "To normalise the input data using the mean/std from the whole 2020 dataset use, norm = True\n"
        "To reduce to minimal output components, minimal_outputs = True \n"
    )
    citation = "-"
    terms_of_use = (
        "By downloading data from this dataset, you agree to the terms and conditions defined at "
        "https://apps.ecmwf.int/datasets/licences/general/ "
        "If you do not agree with such terms, do not download the data. "
    )

    def __init__(
        self,
        subset=None,
        timestep=0,
        filenum=[0],
        input_fields=[
            "sca_inputs",
            "col_inputs",
            "hl_inputs",
            "inter_inputs",
            "pressure_hl",
        ],
        output_fields=["sw", "lw", "hr_sw", "hr_lw"],
        minimal_outputs = False,
        topnetflux = False,
        netflux = False,
        norm=None,
        path=None,
    ):
        self.valid_subset = [
            "tier-1",
            "2020",
            "2019013100",
            "2019053100",
            "2019082900",
            "2019102800",
        ]
        self.valid_timestep = list(range(0, 3501, 125)) + [
            2019013100,
            2019053100,
            2019082900,
            2019102800,
        ]
        self.valid_filenum = list(range(52))
        if subset is not None:
            self.check_valid(self.valid_subset, subset)
            self.timestep = timestep_subset[subset]
            self.filenum = filenum_subset[subset]
        else:
            self.timestep = timestep
            self.filenum = filenum
        self.check_valid(self.valid_timestep, timestep)
        if (type(self.timestep) == int and self.timestep > 2019000000) or (
            type(self.timestep) == list and self.timestep[0] > 2019000000
        ):
            self.valid_filenum = list(range(116))
        self.check_valid(self.valid_filenum, filenum)
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.netflux = netflux
        self.topnetflux = topnetflux
        if path is None:
            request = dict(timestep=self.timestep, url=URL, filenum=self.filenum)
            self.source = cml.load_source(
                "url-pattern", PATTERN, request
            )  # , merger=Merger())
        else:
            if path[-1] == "/":
                path = path[:-1]
            request = dict(timestep=self.timestep, url=path, filenum=self.filenum)
            self.source = cml.load_source(
                "file-pattern", PATTERN, request
            )  # , merger=Merger())

        self.g_cp = tf.constant(9.80665 / 1004)
        assert not (minimal_outputs and self.netflux), "Minimal outputs not consistent with netflux"
        if minimal_outputs:
            if self.topnetflux:
                self.sparsefunc = self.sparsen_topnet
            else:
                self.sparsefunc = self.sparsen_data
            self.sparse_outputs = Intersection(["sw", "lw"], self.output_fields)
        elif self.netflux:
            self.sparse_outputs = Intersection(["sw", "lw"], self.output_fields)            
            self.sparsefunc = self.netflux_data
        else:
            self.sparsefunc = self.emptyfunction

        if norm is None or (norm is False):
            self.normfunc = self.emptyfunction
        else:
            self.input_means, self.input_stds = self.load_norms()
            self.normfunc = self.normalise

    def check_valid(self, valid, inputs):
        if type(inputs) == list:
            for item in inputs:
                assert item in valid, f"{item} not in {valid}"
        else:
            assert inputs in valid, f"{inputs} not in {valid}"
        return

    def emptyfunction(self, x, y):
        return x, y

    def norm_field(self, key, field):
        return (field - self.input_means[key]) / self.input_stds[key]

    def normalise(self, inputs, outputs):
        for k in inputs:
            inputs[k] = self.norm_field(k, inputs[k])
        return inputs, outputs

    def sparsen_data(self, inputs, outputs):
        for k in self.sparse_outputs:
            outputs[k] = tf.stack(
                [outputs[k][..., 0, 1], outputs[k][..., -1, 0], outputs[k][..., -1, 1]],
                axis=-1,
            )
        return inputs, outputs

    def netflux_data(self, inputs, outputs):
        for k in self.sparse_outputs:
            outputs[k] = outputs[k][...,0] - outputs[k][...,1]
        return inputs, outputs

    def sparsen_topnet(self, inputs, outputs):
        for k in self.sparse_outputs:
            outputs[k] = tf.stack(
                [outputs[k][..., 0, 0] - outputs[k][..., 0, 1], outputs[k][..., -1, 0], outputs[k][..., -1, 1]],
                axis=-1,
            )
        return inputs, outputs

    def _parse_batch(self, record_batch):
        # Create a description of the features
        global feature_description
        # Parse the input `tf.Example` proto using the dictionary above
        example = tf.io.parse_example(record_batch, feature_description)
        inputs = {}
        outputs = {}
        for k in self.input_fields:
            inputs[k] = example[k]
        for k in self.output_fields:
            if k in ["hr_sw", "hr_lw"]:
                outputs[k] = self.g_cp * example[k]
            else:
                outputs[k] = example[k]
        return self.sparsefunc(*self.normfunc(inputs, outputs))

    def to_tfdataset(
        self, batch_size=256, shuffle=True, shuffle_size=2048 * 16, repeat=False
    ):
        ds = self.source.to_tfdataset(num_parallel_reads=AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(shuffle_size)
        # Prepare batches
        ds = ds.batch(batch_size)

        # Parse a batch into a dataset
        ds = ds.map(lambda x: self._parse_batch(x))

        if repeat:
            ds = ds.repeat()

        return ds.prefetch(buffer_size=AUTOTUNE)

    def load_norms(self, dataset=None):
        input_means = {}
        input_stds = {}
        if dataset is None:
            print("Loading normalisation arrays")
            request = dict(url=HEADURL, input_fields=self.input_fields)
            tmp_source = self.source
            self.source = cml.load_source(
                "url-pattern", NORM_PATTERN, request, merger=NormMerger()
            )
            dataset = self.to_xarray()
            self.source = tmp_source

        for k in self.input_fields:
            array = dataset[f"{k}_mean"].values
            input_means[k] = tf.constant(array, shape=(1,) + array.shape)
            array = dataset[f"{k}_std"].values
            input_stds[k] = tf.constant(array, shape=(1,) + array.shape)
        return input_means, input_stds


# class Merger:
#     def __init__(self):
#         return

#     def to_tfdataset(self, paths):
#         files_ds = tf.data.Dataset.list_files(paths)
#         ignore_order = tf.data.Options()
#         ignore_order.experimental_deterministic = False
#         files_ds = files_ds.with_options(ignore_order)
#         return tf.data.TFRecordDataset(files_ds, num_parallel_reads=AUTOTUNE)

# This class merges the normalisation files, which have clashing variable names
# which we rename before merging
class NormMerger:
    def __init__(self, engine="netcdf4", options=None):
        self.engine = engine
        self.options = options if options is not None else {}

    def to_xarray(self, paths, **kwargs):
        return xr.open_mfdataset(
            paths,
            engine=self.engine,
            combine="nested",
            coords="minimal",
            data_vars="minimal",
            preprocess=preprocess_norm,
            compat="override",
            parallel=True,
            **self.options,
        )


def preprocess_norm(ds):
    global _num_norm
    _num_norm += 1
    return ds.rename({"variable": f"variable{_num_norm}"})


def Intersection(lst1, lst2):
    return set(lst1).intersection(lst2)
