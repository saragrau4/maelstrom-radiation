## maelstrom-radiation

A dataset plugin for climetlab (https://climetlab.readthedocs.io/en/latest/)
 for the dataset maelstrom-radiation.

## Datasets description

This data is for learning the
emulation of the ECMWF radiation scheme, TripleClouds, found in the ecRad package 
(https://github.com/ecmwf/ecrad). Building an accurate emulator of radiative heating
could accelerate weather and climate models partially by enabling the use of GPU
hardware within our models.

There are two datasets, allowing different views on the same data:

### 1 : `maelstom-radiation`
Supports the `to_xarray` method and allows users to explore the data with all structure kept intact.

### 2 : `maelstrom-radiation-tf`
Loads the same data but from a shuffled and repacked into the TFRecord format. This dataset supports 
`to_tfdataset` which uses Tensorflow's dataset object.


## Using climetlab to access the data 

Both datasets and downloaded and explained in the demo notebook here
https://git.ecmwf.int/projects/MLFET/repos/maelstrom-radiation/browse/notebooks/demo_radiation.ipynb

The climetlab python package allows easy access to the data with a few lines of code such as:
```

!pip install climetlab climetlab_maelstrom_radiation
import climetlab as cml
cml_ds = cml.load_dataset("maelstrom-radiation", subset="tier-1")
ds = cml_ds.to_xarray()

!or for the TFdataset version
cml_ds = cml.load_dataset("maelstrom-radiation-tf", subset="tier-1")
ds = cml_ds.to_tfdataset(batch_size=256)
```
