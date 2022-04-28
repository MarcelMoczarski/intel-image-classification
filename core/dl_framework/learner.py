import pandas as pd
import torch
from core.dl_framework.loss_functions import loss_function
from core.dl_framework.callbacks import get_callbackhandler
from core.dl_framework.model import get_model
from tqdm import tqdm
import torch


class Container:
    def __init__(self, data, config_file):
        self.opt = config_file["g_optimizer"]

        self.bs = config_file["h_batch_size"]
        self.arch = [
            config_file["g_arch"],
            config_file["g_arch_depth"],
            config_file["g_hidden_layers"],
        ]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.lr = config_file["h_lr"]
        self.gpu = config_file["m_gpu"]

        self.data = data
        self.model, self.opt = get_model(
            self.data, self.arch, self.lr, self.opt, self.device
        )

        self.loss_func = loss_function(config_file, self.model)
        
        self.do_stop = False
        self.resume = config_file["g_resume"]


class Learner:
    def __init__(self, data, config_file):
        self.learn = Container(data, config_file)
        self.cbh = get_callbackhandler(config_file, self.learn)
        self.device = self.learn.device

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

    def all_batches(self, data):
        pbar = tqdm(data, total=len(data))
        for batch in pbar:
            self.one_batch(batch)
            self.cbh.on_batch_end()

    def one_batch(self, batch):
        xb, yb, idx = batch
        xb, yb = xb.to(self.learn.device), yb.to(self.learn.device)
        out = self.learn.model(xb)
        loss = self.learn.loss_func.calc(out, yb)
        if not self.cbh.on_loss_end(loss, out, yb):
            return
        loss.backward()
        self.learn.opt.step()
        self.learn.opt.zero_grad()

    @property
    def history(self):
        return pd.DataFrame(self.learn.history_raw).set_index("epochs")
