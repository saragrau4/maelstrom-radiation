from time import time

# To hide GPU uncomment
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)
from tensorflow.keras.optimizers import Adam

from data import load_data
from models import load_model
from pprint import pprint
import plotting


def lw_inference(
    batch_size=256,
    synthetic_data=False,
    model_path="model.h5",
    run_no=0,
    tier=1,
    no_stats = False,
    minimal = False,
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
        output_fields=["lw", "hr_lw"],
    )
    print("Data loaded")
    load_time = time() - total_start

    print(f"Loading model {model_path}")
    model = load_model(model_path)

    gen_stats = (not no_stats)
    if gen_stats:
        for i in range(2):
            eval_start = time()
            eval = model.evaluate(
                test,
                verbose=2,
            )   
            results = dict(zip(model.metrics_names, eval))
            eval_time = time() - eval_start
            total_time = time() - total_start
            # printstats(eval)
            results["load_time"] = load_time
            results["eval_time"] = eval_time
            results["total_time"] = total_time
            pprint(results)

    print("Making plots")
    test_batch = load_data(
        mode="test",
        batch_size=4,
        synthetic_data=synthetic_data,
        cache=False,
        minimal=False,
        tier=tier,
        shuffle=False,
        output_fields=["lw", "hr_lw"],
    ).take(1)
    for inputs, outputs in test_batch:
        pred = model.predict(inputs)

    plotting.plotbatch_wc(
        outputs,
        pred,
        to_plot=["lw_dn","lw_up","hr_lw"],
        save_as=f"columns_{run_no}.png",
    )


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
        "--no_stats",
        help="Don't calculate stats, just do plots",
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
    sw_inference(
        batch_size=args.batch,
        synthetic_data=args.synthetic_data,
        model_path=args.model_path,
        run_no=args.runno,
        tier=args.tier,
        no_stats = args.no_stats
    )
