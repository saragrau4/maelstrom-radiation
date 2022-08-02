import numpy as np
import climetlab as cml

cml.settings.set("check-out-of-date-urls", False)

norms = np.load("inp_max_norm.npy", allow_pickle=True)
norms = norms[()]
for key in norms:
    norms[key][norms[key] == 0] = 1


def load_data(
    mode="train",
    batch_size=256,
    minimal=True,
    synthetic_data=False,
    cache=False,
    tier=1,
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


    assert tier in [1,2,3], f"Tier {tier} not supported"

    print("Climetlab cache dir")
    print(cml.settings.get("cache-directory"))

    tiername = f"tier-{tier}"
    assert mode in ["train","val","test"], f"{mode} not train/val/test"
    
    if mode in ["val","test"]:
        tiername = tiername + f"-{mode}"

    ds_cml = cml.load_dataset(
        "maelstrom-radiation-tf", subset = tiername, **kwargs
    )

    train_num = ds_cml.numcolumns // shard_num
    train = ds_cml.to_tfdataset(
        batch_size=batch_size,
        shuffle=True,
        shard_num=shard_num,
        shard_idx=shard_idx,
        cache=cache,
    )

    if synthetic_data:
        print(
            "Creating synthetic data by repeating single batch, useful for pipeline testing."
        )
        train = train.take(1).cache().repeat(train_num)

    return train


def load_train_val_data(
    **kwargs
):
    train = load_data(mode="train",**kwargs)
    val = load_data(mode="val",**kwargs)
    return train, val