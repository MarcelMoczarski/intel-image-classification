from pathlib import Path
import toml
from core.dl_framework.utils import read_config
from core.dl_framework.data import get_dataset, Dataset, get_dls, split_data, DataBunch
from core.dl_framework import loss_functions
import torch
import core.dl_framework.model as models
from tqdm import tqdm

bs = 64
lr = 1e-05
valid_split = 0.2
loss_func = getattr(loss_functions, "cross_entropy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
arch = "Model_1"

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

def get_model(data, arch, lr, c):
    input_shape = data.train_ds.x.shape[1]
    net = getattr(models, arch)(input_shape, c)
    return net, torch.optim.Adam(net.parameters(), lr = lr)

def fit(epochs, data, model, device, opt):
    for epoch in range(epochs):
        model.train()
        all_batches(data.train_dl, model, device, opt)
        model.eval()
        with torch.no_grad():
            all_batches(data.valid_dl, model, device, opt)

def all_batches(data, model, device, opt):
    pbar = tqdm(data, total=len(data))
    for batch in pbar:
        one_batch(batch, model, device, opt)

def one_batch(batch, model, device, opt):
    xb, yb = batch
    xb, yb = xb.to(device), yb.to(device)
    out = model(xb)
    loss = loss_func(out, yb)
    if not model.training: return 
    loss.backward()
    opt.step()
    opt.zero_grad()




config_path = "./configs/default_train_config.toml"
setup_config = read_config(toml.load(config_path))

x_train, y_train, x_test, y_test = get_dataset(setup_config["s_source"], setup_config["s_set"], setup_config["p_tmp_data_path"])
train_ds, test_ds = Dataset(x_train, y_train), Dataset(x_test, y_test)
data = get_databunch(train_ds, bs, valid_split, num_classes)

model, opt = get_model(data, arch, lr, num_classes)

model = model.to(device)

fit(10, data, model, device, opt)


# 'm_gpu': False,
#  'p_tmp_data_path': './tmp_files',
#  'g_num_epochs': 100,
#  'g_num_classes': 10,
#  'g_valid_split': 0.2,
#  'g_loss_func': 'cross_entropy',
#  'g_optimizer': 'adam',
#  'g_arch': 'Model_2',
#  's_source': 'torchvision',
#  's_set': 'MNIST',
#  'h_batch_size': 64,
#  'h_lr': 1e-05,
#  'c_m_monitor': ['valid_acc', 'valid_loss', 'loss'],
#  'c_m_print2console': True,
#  'c_e_monitor': 'valid_loss',
#  'c_e_patience': 1000}