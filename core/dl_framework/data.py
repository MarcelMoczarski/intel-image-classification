import os
import shutil
from pathlib import Path

import numpy as np
import torchvision
from sklearn.model_selection import train_test_split


def get_dataset(setup_config, filespath=None):
    source, dataset = setup_config["s_source"], setup_config["s_set"]
    save_path, kaggle_json_path = setup_config["p_tmp_data_path"], setup_config["p_kaggle_json_path"]
    CNN = setup_config["g_CNN"]
    build_set_from_folder = setup_config["s_build_set_from_folder"]
    shuffle = setup_config["s_shuffle"]

    
    if source == "torchvision":
        dataset = getattr(torchvision.datasets, dataset)
        train_set = dataset(save_path, train=True, download=True)
        test_set = dataset(save_path, train=False, download=True)
        x_train, y_train = train_set.data / 255, train_set.targets
        x_test, y_test = test_set.data / 255, test_set.targets

        if not CNN:
            x_train = x_train.reshape((len(x_train), -1))
            x_test = x_test.reshape((len(x_test), -1))
            
        return x_train, y_train, x_test, y_test

    if source == "kaggle":
        kaggle_json_path += "/kaggle.json"
        root_path = Path("/root/.kaggle")
        root_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(kaggle_json_path, root_path/"kaggle.json")
        if len(os.listdir(save_path)) <= 1:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(dataset, path=save_path, unzip=True)
        if filespath:
            data = ImageFolder(filespath)
        
        return 0, 0, 0, 4
    

class Dataset():
    # transforms?
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        if self.transform:
            self.x = self.transform(x)
        return self.x[i], self.y[i]


class DataLoader():
    # shuffle? num_workers?
    def __init__(self, ds, bs, drop_last=True):
        self.dataset = ds
        self.bs = bs
        self.drop_last = drop_last
        self._drop_length = len(self.dataset) - bs * \
            int(len(self.dataset) // bs)

    def __len__(self):
        if self.drop_last:
            length = np.floor(len(self.dataset) / self.bs)
        else:
            length = np.ceil(len(self.dataset) / self.bs)
        return int(length)

    def __iter__(self):
        if self.drop_last:
            length = len(self.dataset) - self._drop_length
        else:
            length = len(self.dataset)
        for i in range(0, length, self.bs):
            yield self.dataset[i:i+self.bs]


class DataBunch():
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c

    @property  # setter getter etc. -> instead of DataBunch.train_ds() you can use DataBunch.train_ds to get Dataset
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset


def get_dls(train_ds, valid_ds, bs):
    return DataLoader(train_ds, bs), DataLoader(valid_ds, bs)


def split_data_raw(x_data, y_data, valid_split, stratify=False):
    if stratify == False:
        train_indices, valid_indices, _, _ = train_test_split(
            range(len(x_data)), y_data, test_size=valid_split, random_state=1)
    else:
        train_indices, valid_indices, _, _ = train_test_split(
            range(len(x_data)), y_data, test_size=valid_split, stratify=y_data)
    x_train, y_train = x_data[train_indices], y_data[train_indices]
    x_valid, y_valid = x_data[valid_indices], y_data[valid_indices]
    return Dataset(x_train, y_train), Dataset(x_valid, y_valid)
    # use of subsets here, to safe  ram, if needed

def split_data(data, split_size, stratify=False):
    data = data[0]
    if type(data) == Dataset:
        data = split_data_raw(data.x, data.y, split_size)
    if type(data) == DataLoader:
        data = split_data_raw(data.dataset.x, data.dataset.y, split_size)
    return data

def get_databunch(data, bs, split_size, c):
    if type(data) != list:
        data = [data]
    if len(data) < 2:
        data = split_data(data, split_size)
    else:
        if any(dl for dl in data if type(dl) == DataLoader):
            data = [data[0].dataset, data[1].dataset]
    data = DataBunch(*get_dls(data[0], data[1], bs), c)
    return data

        
