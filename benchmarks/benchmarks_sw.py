from time import time

# To hide GPU uncomment
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from utils import EpochTimingCallback, printstats  # TimingCallback,
from data import load_train_val_data
from models import build_cnn, build_fullcnn, build_rnn, load_model
import losses
import horovod.keras as hvd
#import mlflow.tensorflow

def main(
    batch_size=256,
    epochs=5,
    synthetic_data=False,
    cache=True,
    data_only=False,
    run_no=0,
    model_type="min_cnn",
    tier=1,
    continue_model="",
    attention=False,
):

    # Horovod: initialize Horovod.
    hvd.init()

    if model_type == "min_cnn":
        minimal = True
    else:
        minimal = False

    # Pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    print("Getting training/validation data")
    total_start = time()
    train,val = load_train_val_data(
        batch_size=batch_size,
        synthetic_data=synthetic_data,
        cache=cache,
        minimal=minimal,
        tier=tier,
        shard_num=hvd.size(),
        shard_idx=hvd.local_rank(),
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
        )
        loss = {"hr_sw": "mse", "sw": "mse"}
        weights = {"hr_sw": 10 ** (3), "sw": 1}
        lr = 10 ** (-4)
    elif model_type == "rnn":
        model = build_rnn(
            train.element_spec[0],
            train.element_spec[1],
        )
        loss = {"hr_sw": "mae", "sw": losses.top_scaledflux_mae}
        weights = {"hr_sw": 10 ** (-1), "sw": 1}
        lr = 0.5 * 10 ** (-3)
    elif model_type == "cnn":
        model = build_fullcnn(
            train.element_spec[0],
            train.element_spec[1],
            attention=attention,
        )
        loss = {"hr_sw": "mae", "sw": losses.top_scaledflux_mae}
        weights = {"hr_sw": 10 ** (-1), "sw": 1}
        lr = 2 * 10 ** (-4)
    else:
        assert False, f"{model_type} not configured"
    if continue_model is not "":
        print(f"Continuing {continue_model}")
        model = load_model(continue_model)

    # Horovod: add Horovod Distributed Optimizer.
    opt = Adam(lr * batch_size / 256 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)

    model.compile(
        loss=loss,
        metrics={"hr_sw": ["mse", "mae"], "sw": ["mse", "mae"]},
        loss_weights=weights,
        optimizer=opt,
        experimental_run_tf_function=False,
    )
    model.summary()

    logfile = f"training_{model_type}_{run_no}.log"
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
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
    if hvd.rank() == 0:
        callbacks.append(EpochTimingCallback())
        callbacks.append(tf.keras.callbacks.CSVLogger(logfile))
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                f"./{model_type}-{run_no}" + "-{epoch}.h5"
            )
        )

    train_start = time()
    _ = model.fit(
        train,
        validation_data=val,
        epochs=epochs,
        verbose=2 if hvd.rank() == 0 else 0,
        callbacks=callbacks,
    )
    train_time = time() - train_start
    save_start = time()
    if hvd.rank() == 0:
        model.save(f"{model_type}_{run_no}.h5")
    save_time = time() - save_start
    total_time = time() - total_start

    printstats(logfile, total_time, train_time, load_time, save_time, batch_size)


if __name__ == "__main__":
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
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model", type=str, default="min_cnn")
    parser.add_argument("--continue_model", type=str, default="")
    parser.add_argument(
        "--nocache",
        help="Don't cache dataset",
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
        attention=args.attention
    )
