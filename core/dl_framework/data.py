"""_summary_

Returns:
    _type_: _description_
"""
from __future__ import annotations

import os
import shutil
import sys
import typing
from pathlib import Path
import numpy as np
from PIL import Image

# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


def download_data(setup_config: typing.Dict[str, typing.Any]) -> None:
    if setup_config["s_source"] == "kaggle":
        if len(os.listdir(setup_config["p_tmp_data_path"])) <= 1:
            kaggle_json_file: str = setup_config["p_kaggle_json_path"] + "/kaggle.json"
            root_path = Path("/root/.kaggle")
            root_path.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(kaggle_json_file, root_path / "kaggle.json")

            import kaggle

            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                setup_config["s_set"],
                path=setup_config["p_tmp_data_path"],
                unzip=True,
            )


def data_pipeline(rel_data_path: str, config_file: typing.Dict) -> DataBunch:
    """pipeline that creates databunch from files in rel_data_path and with parameters specified in configfile

    Args:
        rel_data_path (str): path to data
        config_file (typing.Dict): config file

    Returns:
        DataBunch: returns databunch containing train and valid dataloaders
    """
    data_path = Path(config_file["p_tmp_data_path"]) / Path(rel_data_path)
    data_ds = CustomDataset(data_path, get_transforms(config_file))

    num_classes = len(data_ds.classes_to_label)

    train_idx, valid_idx = get_samplers(data_ds.targets, config_file["g_valid_size"], stratify=True)
    train_dl = DataLoader(
        data_ds, batch_size=config_file["h_batch_size"], sampler=train_idx, drop_last=True
    )
    valid_dl = DataLoader(data_ds, batch_size=config_file["h_batch_size"], sampler=valid_idx)

    data = DataBunch(train_dl, valid_dl, num_classes)
    return data


def get_samplers(
    targets: np.ndarray, valid_size: float, stratify: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Returns train and valid indices

    Args:
        targets (np.ndarray): target array
        valid_size (float): split size
        stratify (bool, optional): stratify if data has class imbalances. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: returns shuffled indices vor train and valid loaders
    """
    if stratify:
        train_idx, valid_idx = train_test_split(
            range(len(targets)),
            shuffle=True,
            test_size=valid_size,
            stratify=targets,
            random_state=1,
        )
    else:
        train_idx, valid_idx = train_test_split(
            range(len(targets)), shuffle=True, test_size=valid_size, random_state=1
        )

    return np.array(train_idx), np.array(valid_idx)


def get_transforms(config_file: typing.Dict) -> list:
    """gets list of transforms from configfile

    Args:
        config_file (typing.Dict): configfile

    Returns:
        list: list containing all transforms specified in configfile
    """
    config_transforms = [
        (key.split("_", 2)[-1], value) for key, value in config_file.items() if "s_t_" in key
    ]
    transform = []
    for t in config_transforms:
        if t[1]:
            transform.append(getattr(transforms, t[0])(t[1]))
        else:
            transform.append(getattr(transforms, t[0])())
    return transform


class CustomDataset(Dataset):
    """Dataset class for loading jpg files

    Args:
        CustomDataset (Dataset): data_path, transform list
    """

    def __init__(self, data_path: typing.Union[str, Path], transform: list = None) -> None:
        self.data_path = data_path
        self.transform = transform
        self.x = sorted([x for x in Path(data_path).rglob("*.jpg") if x.is_file()])
        y_classes = [y.name for y in Path(data_path).glob("*")]
        self.classes_to_label = dict(zip(y_classes, range(len(y_classes))))
        self.targets = np.array([self.classes_to_label[x.parent.name] for x in self.x])

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple:
        xs = Image.open(self.x[idx])
        if self.transform:
            xs = transforms.Compose(self.transform)(xs)
        ys_class = self.x[idx].parent.name
        ys = self.classes_to_label[ys_class]
        return xs, ys


class DataBunch:
    """databunch class is bucket for train and valid dataloaders and for number of classes"""

    def __init__(self, train_dl: DataLoader, valid_dl: DataLoader, c: int) -> None:
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c


# class DataBunch():
#     def __init__(self, train_dl, valid_dl, c=None):
#         self.train_dl = train_dl
#         self.valid_dl = valid_dl
#         self.c = c

#     @property  # setter getter etc. -> instead of DataBunch.train_ds() you can use DataBunch.train_ds to get Dataset
#     def train_ds(self):
#         return self.train_dl.dataset

#     @property
#     def valid_ds(self):
#         return self.valid_dl.dataset


# def get_dls(train_ds, valid_ds, bs):
#     return DataLoader(train_ds, bs), DataLoader(valid_ds, bs)


# def split_data_raw(x_data, y_data, valid_split, stratify=False):
#     if stratify == False:
#         train_indices, valid_indices, _, _ = train_test_split(
#             range(len(x_data)), y_data, test_size=valid_split, random_state=1)
#     else:
#         train_indices, valid_indices, _, _ = train_test_split(
#             range(len(x_data)), y_data, test_size=valid_split, stratify=y_data)
#     x_train, y_train = x_data[train_indices], y_data[train_indices]
#     x_valid, y_valid = x_data[valid_indices], y_data[valid_indices]
#     return Dataset(x_train, y_train), Dataset(x_valid, y_valid)
#     # use of subsets here, to safe  ram, if needed

# def split_data(data, split_size, stratify=False):
#     data = data[0]
#     if type(data) == Dataset:
#         data = split_data_raw(data.x, data.y, split_size)
#     if type(data) == DataLoader:
#         data = split_data_raw(data.dataset.x, data.dataset.y, split_size)
#     return data

# def get_databunch(data, bs, split_size, c):
#     if type(data) != list:
#         data = [data]
#     if len(data) < 2:
#         data = split_data(data, split_size)
#     else:
#         if any(dl for dl in data if type(dl) == DataLoader):
#             data = [data[0].dataset, data[1].dataset]
#     data = DataBunch(*get_dls(data[0], data[1], bs), c)
#     return data
