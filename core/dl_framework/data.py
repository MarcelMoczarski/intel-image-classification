import os
import shutil
from pathlib import Path
from torchvision import transforms
from PIL import Image
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset, DataLoader

def download_data(setup_config: dict) -> None:
    if setup_config["s_source"] == "kaggle":
        kaggle_json_file = setup_config["p_kaggle_json_path"] +  "/kaggle.json"
        root_path = Path("/root/.kaggle")
        root_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(kaggle_json_file, root_path/"kaggle.json")
        if len(os.listdir(setup_config["p_tmp_data_path"])) <= 1:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(setup_config["s_set"], path=setup_config["p_tmp_data_path"], unzip=True)

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=[]) -> None:
        self.data_path = data_path
        self.transform = transform
        self.x = sorted([x for x in Path(data_path).rglob("*.jpg") if x.is_file()])
        y_classes = [y.name for y in Path(data_path).glob("*")]
        self.classes_to_label = dict(zip(y_classes, range(len(y_classes))))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        xs = Image.open(self.x[idx])
        if self.transform:
            xs = transforms.Compose(self.transform)(xs)
        ys_class = self.x[idx].parent.name
        ys = self.classes_to_label[ys_class]
        return xs, ys

class DataBunch():
    def __init__(self, train_dl, valid_dl, c) -> None:
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c
    
    @property
    def train_ds(self) -> DataLoader:
        return self.train_dl
    


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

        
