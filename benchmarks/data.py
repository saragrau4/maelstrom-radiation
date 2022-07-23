import numpy as np
import climetlab as cml

cml.settings.set("check-out-of-date-urls", False)

norms = np.load("inp_max_norm.npy", allow_pickle=True)
norms = norms[()]
for key in norms:
    norms[key][norms[key] == 0] = 1
norms


def load_data(
    batch_size=256,
    minimal=True,
    sample_data=False,
    synthetic_data=False,
    cache=False,
    version=1,
    shard_num=1,
    shard_idx=1,
):

    kwargs = {
        "hr_units": "K d-1",
        "norm": False,
        "dataset": "tripleclouds",
        "output_fields": ["sw", "hr_sw"],
    }
    if minimal:
        kwargs["minimal_outputs"] = True
        kwargs["topnetflux"] = True
    else:
        kwargs["minimal_outputs"] = False

    if sample_data:
        train_ts = [0]
        train_fn = [0]
        val_ts = 2019013100
        val_fn = [0]
    elif version == 1:
        train_ts = list(range(0, 3501, 1000))  # 500))
        train_fn = list(range(0, 51, 5))
        val_ts = [2019013100, 2019082900]
        val_fn = [0, 25, 50] 
    elif version == 2:
        train_ts = list(range(0, 3501, 125))  # 500))
        train_fn = list(range(0, 51, 5))
        val_ts = [2019013100, 2019082900]
        val_fn = [0, 25, 50]  
    elif version == 3:
        train_ts = list(range(0, 3501, 250))  # 500))
        train_fn = list(range(0, 51, 5))
        val_ts = [2019013100, 2019082900]
        val_fn = [0, 25, 50] 
    else:
        assert False, f"Version {version} not supported"

    train_num = 67840 * len(train_ts) * len(train_fn) // shard_num
    val_num = 67840 * len(val_ts) * len(val_fn) // shard_num

    print("Climetlab cache dir")
    print(cml.settings.get("cache-directory"))

    ds_cml = cml.load_dataset(
        "maelstrom-radiation-tf", timestep=train_ts, filenum=train_fn, **kwargs
    )

    train = ds_cml.to_tfdataset(batch_size=batch_size, shuffle=True,
        shard_num = shard_num, shard_idx = shard_idx,   
    ) 

    ds_cml_val = cml.load_dataset(
        "maelstrom-radiation-tf", timestep=val_ts, filenum=val_fn, **kwargs
    )

    val = ds_cml_val.to_tfdataset(batch_size=batch_size, shuffle=False,
        shard_num = shard_num, shard_idx = shard_idx, 
    )

    if synthetic_data:
        print(
            "Creating synthetic data by repeating single batch, useful for pipeline testing."
        )
        train = train.take(1).cache().repeat(train_num)
        val = val.take(1).cache().repeat(val_num)
    elif cache:
        print("Caching dataset, increase memory use, decreased runtime")
        train = train.cache()
        val = val.cache()

    return train, val
