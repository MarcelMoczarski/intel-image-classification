from pathlib import Path
from typing import Optional
import click
import toml
# from core.dl_framework.data import download_data
# from core.dl_framework.learner import Learner
from core.dl_framework.utils import read_config as read_config
import torch

@click.command()
@click.argument("config_path", default="./configs/default_train_config.toml", type=click.Path(exists=True))

def main(config_path: str) -> None:
    torch.manual_seed(1)
    read_config("s")
    # setup_config = read_config(toml.load(config_path))
    # download_data(setup_config)
    print("hjoi")
    
    # x_train, y_train, x_test, y_test = get_dataset(setup_config)

    #train_ds, test_ds = Dataset(x_train, y_train), Dataset(x_test, y_test)

    #learn = Learner(train_ds, setup_config)

    # learn = Learner(train_ds, setup_config)
    
    #learn.fit(setup_config["g_num_epochs"])
    

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