#!/usr/bin/env python
# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot

longnames = {
    "hr_sw": "Shortwave heating \n rate (K/d)",
    "hr_lw": "Longwave heating \n rate (K/d)",
    "sw": "Shortwave fluxes (W/m2)",
    "lw": "Longwave fluxes (W/m2)",
    "sw_dn": "Shortwave downwards \n fluxes (W/m2)",
    "sw_up": "Shortwave upwards \n fluxes (W/m2)",
    "lw_dn": "Longwave downwards \n fluxes (W/m2)",
    "lw_up": "Longwave upwards \n fluxes (W/m2)",
}

list2dic = {"hr_sw": 1, "hr_lw": 0, "sw": 3, "lw": 2}


def get_xgrid(required_levels, batch_size, pressure_half_levels=None):
    if pressure_half_levels is None:
        pressure_levels = np.broadcast_to(
            np.arange(1, required_levels + 1), shape=(batch_size, required_levels)
        )
        widths = np.ones(shape=(batch_size, required_levels))
    else:
        if required_levels == 137:
            pressure_levels = (
                (pressure_half_levels[:, 1:, 0] + pressure_half_levels[:, :-1, 0]) / 2
            ).numpy() / 1000
            widths = (
                (pressure_half_levels[:, 1:, 0] - pressure_half_levels[:, :-1, 0])
            ).numpy() / 1000
        else:
            pressure_levels = pressure_half_levels[..., 0].numpy() / 1000
            widths = pressure_half_levels[..., 0].numpy() / 1000
    return pressure_levels, widths


def plotbatch_wc(
    batch_out,
    pred_batch,
    save_as="columns.png",
    to_plot=["sw", "lw", "hr_sw", "hr_lw"],
    cloud_frac=None,
    pressure_half_levels=None,
):
    for key in batch_out:
        batch_size = batch_out[key].shape[0]
        break

    plt.figure(figsize=(10, 2 * len(to_plot) + 0.5), dpi=400)
    # plt.suptitle("Validation example columns")
    columns = np.arange(batch_size)
    for i, key in enumerate(to_plot):
        for j, k in enumerate(columns):
            host1 = host_subplot(len(to_plot), batch_size, batch_size * i + j + 1)
            par1 = host1.twinx()
            host1.yaxis.set_visible(False)

            if key[-2:] in ["dn", "up"]:
                idx = 0 if key[-2:] == "dn" else 1
                truth = batch_out[key[:-3]][k, :, idx]
                if type(pred_batch) == dict:
                    pred = pred_batch[key[:-3]][k, :, idx]
                elif type(pred_batch) == list:
                    pred = pred_batch[list2dic[key[:-3]]][k, :, idx]
            else:
                truth = batch_out[key][k, :]
                if type(pred_batch) == dict:
                    pred = pred_batch[key][k, :]
                elif type(pred_batch) == list:
                    pred = pred_batch[list2dic[key]][k, :]

            pressure_levels, widths = get_xgrid(
                truth.shape[0],
                batch_size,
                pressure_half_levels=pressure_half_levels,
            )

            if cloud_frac is not None:
                host1.bar(
                    x=pressure_levels[k, :],
                    height=cloud_frac[k, :],
                    width=widths[k, :],
                    color="grey",
                )
                host1.set_ylim(0, 1)

            par1.plot(pressure_levels[k, :], truth, label="Truth")
            par1.plot(pressure_levels[k, :], pred, label="Prediction", alpha=0.5)
            par1.yaxis.tick_left()
            par1.yaxis.set_label_position('left')
            ax = plt.gca()
            if i == 0:
                plt.title(f"{k}")
            if i < len(to_plot) - 1:
                ax.axes.xaxis.set_ticklabels([])
            if j == 0:
                par1.set_ylabel(longnames[key])
            # else:
            #    ax.axes.yaxis.set_ticklabels([])
            # if j < 2:
            #    plt.ylim([0,500])
            # else:
            #    plt.ylim([-10,10])
            if (i == len(to_plot) - 1) and (j == batch_size - 1):
                plt.legend()
    plt.subplots_adjust(left=0.075,
                    bottom=0.075, 
                    right=0.95, 
                    top=0.95, 
                    wspace=0.3, 
                    hspace=0.3)
    plt.suptitle("Model Level", x=0.5, y=0.025)
    plt.savefig(save_as, dpi=400)
    #plt.tight_layout()
    return
