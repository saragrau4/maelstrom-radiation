#!/usr/bin/env python
# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from time import time

# To hide GPU uncomment
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import mantik

import tensorflow as tf         # 
from tensorflow.keras.optimizers import Adam

from .utils import EpochTimingCallback, printstats, print_gpu_usage, print_cpu_usage  # TimingCallback,
from .data import load_train_val_data
from .models import build_cnn, build_fullcnn, build_rnn, load_model
from climetlab_maelstrom_radiation.benchmarks import losses
try:
    import horovod.keras as hvd
    print("Horovod found")
    have_hvd=True
except:
    print("No Horovod")
    have_hvd=False

# from deep500.utils import timer_tf as timer
# import mlflow


def main(
    batch_size=256,
    epochs=5,
    synthetic_data=False,
    cache=True,
    data_only=False,
    run_no=0,
    model_type="cnn",
    tier=1,
    continue_model="",
    no_recompile=False,
    attention=False,
    inference=False,
    no_tf32=False,
    dl_test=False,
):

    # mantik.init_tracking()
    # mlflow.tensorflow.autolog()
    print(f"   JOBID is : {run_no}")
    # Horovod: initialize Horovod.
    if have_hvd:
        hvd.init()

    if model_type == "min_cnn":
        minimal = True
    else:
        minimal = False
    if no_tf32:
        tf.config.experimental.enable_tensor_float_32_execution(False)
        print("Turning off TF32 for cnn+attention")
    # Pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        gpu_local_rank = hvd.local_rank() if have_hvd else 0
        gpu_rank = hvd.rank() if have_hvd else 0
        tf.config.experimental.set_visible_devices(gpus[gpu_local_rank], "GPU")

    print("Getting training/validation data")
    total_start = time()
    train, val = load_train_val_data(
        batch_size=batch_size,
        synthetic_data=synthetic_data,
        cache=cache,
        minimal=minimal,
        tier=tier,
        shard_num=hvd.size() if have_hvd else 1,
        shard_idx=gpu_rank,
    )
    print("Data loaded")
    load_time = time() - total_start
    if data_only:
        print("Only loading data, quitting now")
        print("Load time: ", load_time)
        return

    if model_type == "min_cnn":
        model = build_cnn(
            train.element_spec[0],
            train.element_spec[1],
            dl_test=dl_test
        )
        loss = {"hr_sw": "mse", "sw": "mse"}
        weights = {"hr_sw": 10 ** (3), "sw": 1}
        lr = 10 ** (-4)
    elif model_type == "rnn":
        model = build_rnn(
            train.element_spec[0],
            train.element_spec[1],
            dl_test=dl_test
        )
        loss = {"hr_sw": "mae", "sw": losses.top_scaledflux_mae}
        weights = {"hr_sw": 10 ** (-1), "sw": 1}
        lr = 0.5 * 10 ** (-3)
        if have_hvd:
            if hvd.size() == 4:
                lr = lr / 2
    elif model_type == "cnn":
        model = build_fullcnn(
            train.element_spec[0],
            train.element_spec[1],
            attention=attention,
            dl_test=dl_test
        )
        loss = {"hr_sw": "mae", "sw": losses.top_scaledflux_mae}
        weights = {"hr_sw": 10 ** (-1), "sw": 1}
        lr = 2 * 10 ** (-4)
    else:
        assert False, f"{model_type} not configured"

    if len(continue_model) > 0:
        print(f"Continuing {continue_model}")
        model = load_model(continue_model)

    if not no_recompile:
        # Horovod: add Horovod Distributed Optimizer.
        n_gpu = hvd.size() if have_hvd else 1
        true_lr = lr * batch_size / 256 * n_gpu
        opt = Adam(true_lr)
        if have_hvd:
            opt = hvd.DistributedOptimizer(opt)

        model.compile(
            loss=loss,
            metrics={"hr_sw": ["mse", "mae"], "sw": ["mse", "mae"]},
            loss_weights=weights,
            optimizer=opt,
            # experimental_run_tf_function=False,
        )
    else:
        assert len(continue_model) > 0, "Cannot use no_recompile without continue_model"
        if have_hvd:
            assert hvd.size() == 1, "Not tested restarting with > 0 GPU"

    model.summary()

    logfile = f"training_{model_type}_{run_no}.log"
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.25, patience=4, verbose=1, min_lr=10 ** (-6)
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=6,
            verbose=2,
            mode="auto",
            restore_best_weights=True,
        ),
    ]
    if have_hvd:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())

    if (gpu_rank == 0):
        callbacks.append(EpochTimingCallback())
        callbacks.append(tf.keras.callbacks.CSVLogger(logfile))
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                f"./{model_type}_{run_no}_" + "{epoch}.h5",
            )
        )
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                f"./{model_type}_{run_no}.h5",
                save_best_only=True,
            )
        )
        # tmr = timer.CPUGPUTimer()
        # callbacks.append(timer.TimerCallback(tmr, gpu=False))


    train_start = time()
    _ = model.fit(
        train,
        validation_data=val,
        epochs=epochs,
        verbose=2 if gpu_rank == 0 else 0,
        callbacks=callbacks,
    )

    train_time = time() - train_start
    # tmr.print_all_time_stats()
    
    # If rank 1, run inference
    if gpu_rank == 0 and inference:
        from .benchmarks_sw_inference import sw_inference

        sw_inference(
            model_path=f"{model_type}_{run_no}.h5",
            batch_size=batch_size,
            synthetic_data=synthetic_data,
            tier=tier,
            run_no=run_no,
            minimal=minimal,
        )
        
    total_time = time() - total_start

    print(f"   Total runtime: {total_time:.2f} s")
    print(f"   Total training time: {train_time:.2f} s")
    data_vol = [2*1.4,None,50*1.4][tier-1]
    batches = [133*512/batch_size,None,5830*512/batch_size][tier-1]
    print(f"   Average performance: {data_vol / train_time * epochs:.2f} GB/s")
    print(f"   Average time per batch: {total_time / batches / epochs:.2f} s")

    print_gpu_usage("   Final GPU memory: ")
    print_cpu_usage("   Final CPU memory: ")

    # printstats(logfile, total_time, train_time, load_time, 0.0, batch_size)

def benchmarks_sw_wrapper():
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark")
    parser.add_argument(
        "--synthetic_data",
        help="Use synthetic dataset for pipeline testing",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--data_only",
        help="Only load data.",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--attention",
        help="Use attention in CNN.",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--tier",
        help="Dataset version",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--inference",
        help="Run inference on test set",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model", type=str, default="min_cnn")
    parser.add_argument("--continue_model", type=str, default="")
    parser.add_argument(
        "--no_recompile",
        help="Continue model without recompling",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--nocache",
        help="Don't cache dataset",
        action="store_const",
        const=True,
        default=False,
    )

    parser.add_argument(
        "--notf32",
        help="Don't use TensorFloat32",
        action="store_const",
        const=True,
        default=False,
    )

    parser.add_argument(
        "--dl_test",
        help="Test dataloader",
        action="store_const",
        const=True,
        default=False,
    )

    parser.add_argument("--runno", type=int)

    args = parser.parse_args()
    main(
        batch_size=args.batch,
        epochs=args.epochs,
        synthetic_data=args.synthetic_data,
        cache=(not args.nocache),
        data_only=args.data_only,
        run_no=args.runno,
        model_type=args.model,
        tier=args.tier,
        continue_model=args.continue_model,
        no_recompile=args.no_recompile,
        attention=args.attention,
        inference=args.inference,
        no_tf32=args.notf32,
        dl_test=args.dl_test
    )
    
if __name__ == "__main__":
    benchmarks_sw_wrapper()
