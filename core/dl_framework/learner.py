import pandas as pd
import torch
from core.dl_framework import loss_functions
from core.dl_framework.callbacks import get_callbackhandler
from core.dl_framework.data import (DataBunch, DataLoader, Dataset,
                                    get_databunch, get_dls, split_data)
from core.dl_framework.model import get_model
from tqdm import tqdm


class Container():

    def __init__(self, data, setup_config):
        self.opt = setup_config["g_optimizer"]
        self.loss_func = getattr(loss_functions, setup_config["g_loss_func"])
        self.bs = setup_config["h_batch_size"]
        self.arch = setup_config["g_arch"]
        self.c = setup_config["g_num_classes"]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.lr = setup_config["h_lr"]
        self.gpu = setup_config["m_gpu"]
        self.data = get_databunch(
            data, self.bs, setup_config["g_valid_split"], self.c)
        self.model, self.opt = get_model(
            self.data, self.arch, self.lr, self.c, self.opt, self.device)
        self.do_stop = False
        self.resume = setup_config["g_resume"]

class Learner():
    def __init__(self, data, setup_config):
        self.learn = Container(data, setup_config)
        self.cbh = get_callbackhandler(setup_config, self.learn)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, epochs):
        self.cbh.on_train_begin(epochs)
        
        if not self.learn.resume:
            start = 0
        else:
            start = self.learn.history_raw["epochs"][-1]
            
        for epoch in range(start, epochs):
            if self.learn.do_stop:
                break
            self.cbh.on_epoch_begin(epoch)
            self.all_batches(self.learn.data.train_dl)

            self.cbh.on_validate_begin()
            with torch.no_grad():
                self.all_batches(self.learn.data.valid_dl)
            self.cbh.on_validate_end()
            self.cbh.on_epoch_end()

        # self.cbh.on_epoch_begin(epoch)  # change to on_train_end

    def all_batches(self, data):
        pbar = tqdm(data, total=len(data))
        for batch in pbar:
            self.one_batch(batch)
            self.cbh.on_batch_end()

    def one_batch(self, batch):
        xb, yb = batch
        xb = xb.unsqueeze(1)
        xb, yb = xb.to(self.learn.device), yb.to(self.learn.device)
        out = self.learn.model(xb)
        loss = self.learn.loss_func(out, yb)
        if not self.cbh.on_loss_end(loss, out, yb):
            return
        loss.backward()
        self.learn.opt.step()
        self.learn.opt.zero_grad()

    @property
    def history(self):
        return pd.DataFrame(self.learn.history_raw).set_index("epochs")


#     def save(self, path, run, **kwargs):
#         state = {
#             "epoch": self.epoch + 1,
#             "state_dict": self.model.state_dict(),
#             "optimizer": self.opt.state_dict()
#         }
#         path = path / "checkpoint_model.pt"
#         torch.save(state, path)
