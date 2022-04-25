"""placeholder for meaningful docstring"""
import typing
import sys

# from pathlib import Path

import click
import toml
import torch
from core.dl_framework.data import data_pipeline
from tqdm import tqdm
# from core.dl_framework.learner import Learner
from core.dl_framework.utils import read_config


@click.command()
@click.argument(
    "config_path",
    default="./configs/default_train_config.toml",
    type=click.Path(exists=True),
)
@click.option("--img_data_path", "-p", default="/seg_train/seg_train", type=str)
@click.option("--set_name", "-n", default="train_data", type=str)
@click.option("--all_transforms", "-a", default=False, type=bool)

def main(config_path: str, img_data_path: str, set_name: str, all_transforms: bool) -> None:
    """_summary_

    Args:
        config_path (str): _description_
    """
    torch.manual_seed(1)
    config_file = read_config(toml.load(config_path))

    print("load data into memory...")
    data = data_pipeline(config_file, img_data_path, set_name, all_transforms)


    pbar_train_dl = tqdm(data.train_dl, total=len(data.train_dl))
    for i, batch in enumerate(pbar_train_dl):
        i
    # data_pipeline()

    # test: typing.Dict[str, str] = {"as" : 1}
    # test = typing.Dict[str, str]
    # a: test = {"a" : int(2)}
    # b: typing.Dict[str, str] = {"as" : 1}
    # x_train, y_train, x_test, y_test = get_dataset(setup_config)

    # train_ds, test_ds = Dataset(x_train, y_train), Dataset(x_test, y_test)

    # learn = Learner(train_ds, setup_config)

    # learn = Learner(train_ds, setup_config)

    # learn.fit(setup_config["g_num_epochs"])


if __name__ == "__main__":
    main()

# TODO when starting: un- and reinstall mypy, install pylint, install mypy --types-types
# TODO activate interactive interpr. in mypy ext settings, click upgrade, install black


# add telegramlogger
# check pytest
# transferlearning
# gridsearch/... hyperparam opt
# model arch via blocks
# test c ext for faster calc
# correlation matrix
# fastai# roadmap
# torch.ijt -> TorchScript. faster?

# msadfypy: install ext, run daemon: dmypy start, upgrade click to v.8.1.2
