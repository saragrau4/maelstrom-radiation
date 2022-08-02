from time import time

# To hide GPU uncomment
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from utils import EpochTimingCallback, printstats  # TimingCallback,
from data import load_data
from models import build_cnn, build_fullcnn, build_rnn
import losses
import layers
from pprint import pprint


def main(
    batch_size=256,
    synthetic_data=False,
    model_path="model.h5",
    run_no=0,
    tier=1,
):

    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("Getting training/validation data")
    total_start = time()
    test = load_data(
        mode="test",
        batch_size=batch_size,
        synthetic_data=synthetic_data,
        cache=False,
        minimal=False,
        tier=tier,
    )
    print("Data loaded")
    load_time = time() - total_start

    print(f"Loading model {model_path}")
    custom_objects = {
        "top_scaledflux_mse": losses.top_scaledflux_mse,
        "top_scaledflux_mae": losses.top_scaledflux_mae,
        "rnncolumns": layers.rnncolumns_old,
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    eval_start = time()
    eval = model.evaluate(
        test,
        verbose=2,
    )
    results = dict(zip(model.metrics_names,eval))
    eval_time = time() - eval_start
    total_time = time() - total_start
    #printstats(eval)
    results['load_time'] = load_time
    results['eval_time'] = eval_time
    results['total_time'] = total_time
    pprint(results)

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
        "--tier",
        help="Dataset tier",
        type=int,
        default=1,
    )

    parser.add_argument("--model_path", type=str, default="model.h5")
    parser.add_argument("--batch", type=int, default=512)

    parser.add_argument("--runno", type=int)

    args = parser.parse_args()
    main(
        batch_size=args.batch,
        synthetic_data=args.synthetic_data,
        model_path=args.model_path,
        run_no=args.runno,
        tier=args.tier,
    )
