## maelstrom-radiation

A dataset plugin for climetlab for the dataset maelstrom-radiation/radiation.


Features
--------

In this README is a description of how to get the maelstrom-radiation.

## Datasets description

There are two datasets: 

### 1 : `radiation`


### 2
TODO


## Using climetlab to access the data (supports grib, netcdf and zarr)

See the demo notebooks here (https://github.com/ecmwf-lab/climetlab_maelstrom_radiation/notebooks

https://github.com/ecmwf-lab/climetlab_maelstrom_radiation/notebooks/demo_radiation.ipynb
[nbviewer] (https://nbviewer.jupyter.org/github/climetlab_maelstrom_radiation/blob/main/notebooks/demo_radiation.ipynb) 
[colab] (https://colab.research.google.com/github/climetlab_maelstrom_radiation/blob/main/notebooks/demo_radiation.ipynb) 

The climetlab python package allows easy access to the data with a few lines of code such as:
```

!pip install climetlab climetlab_maelstrom_radiation
import climetlab as cml
ds = cml.load_dataset(""maelstrom-radiation-radiation", date='20201231',)
ds.to_xarray()
```
