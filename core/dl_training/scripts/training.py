"""placeholder for meaningful docstring"""
import typing
# from pathlib import Path

import click
import toml
import torch
from core.dl_framework.data import download_data
# from core.dl_framework.learner import Learner
from core.dl_framework.utils import read_config


@click.command()
@click.argument(
    "config_path",
    default="./configs/default_train_config.toml",
    type=click.Path(exists=True))

def main(config_path: str) -> None:
    """_summary_

    Args:
        config_path (str): _description_
    """
    torch.manual_seed(1)
    setup_config: typing.Dict[str, typing.Any] = read_config(
        toml.load(config_path))
    download_data(setup_config)
    # test: typing.Dict[str, str] = {"as" : 1}
    # test = typing.Dict[str, str]
    # a: test = {"a" : int(2)}
    # b: typing.Dict[str, str] = {"as" : 1}
    # x_train, y_train, x_test, y_test = get_dataset(setup_config)

    #train_ds, test_ds = Dataset(x_train, y_train), Dataset(x_test, y_test)

    #learn = Learner(train_ds, setup_config)

    # learn = Learner(train_ds, setup_config)

    # learn.fit(setup_config["g_num_epochs"])
def test_func(arg):
    print(arg)

if __name__ == "__main__":
    main()


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
