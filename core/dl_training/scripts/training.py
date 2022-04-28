"""trainings script"""
import click
import toml
import torch
from core.dl_framework.data import data_pipeline
from core.dl_framework.learner import Learner
from core.dl_framework.utils import read_config


@click.command()
@click.argument(
    "config_path",
    default="./configs/default_train_config.toml",
    type=click.Path(exists=True),
)
@click.option(
    "--img_data_path", "-p", default="/seg_train/seg_train", type=str
)
@click.option("--set_name", "-n", default="train_data", type=str)
@click.option("--all_transforms", "-a", default=False, type=bool)
def main(
    config_path: str, img_data_path: str, set_name: str, all_transforms: bool
) -> None:
    """_summary_

    Args:
        config_path (str): _description_
    """
    torch.manual_seed(1)
    config_file = read_config(toml.load(config_path))
    
    print("load data into memory...")
    data = data_pipeline(config_file, img_data_path, set_name, all_transforms)

    learn = Learner(data, config_file)

    learn.fit(config_file["g_num_epochs"])


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
# gridsearch
