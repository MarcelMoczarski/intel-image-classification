import shutil
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import pytz
import torch
from torch.utils.tensorboard import SummaryWriter

"""all checkpoints should be included at the moment

implemented callbacks:
    Recorder: Tracking train/valid loss and setattr[yb, out, loss] for all child classes
    CudaCallback: Manages devices and where data is send to
    Monitor: Tracks 
"""


class Callback:
    """parent class for all Callbacks
    implements dummy methods
    """

    def __init__(self, learn):
        self.learn = learn

    def on_train_begin(self, epochs):
        self.epochs = epochs

    def on_train_end(self):
        pass

    def on_epoch_begin(self, epoch):
        self.epoch = epoch

    def on_epoch_end(self):
        pass

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self):
        pass

    def on_loss_begin(self):
        pass

    def on_loss_end(self, loss, out, yb):
        self.loss = loss

    def on_step_begin(self):
        pass

    def on_step_end(self):
        pass

    def on_validate_begin(self):
        pass

    def on_validate_end(self):
        self.learn.model.train()


class CallbackHandler:
    def __init__(self, cbs, learn):
        self.learn = learn
        self.cbs = cbs
        for cb in self.cbs:
            setattr(self, type(cb).__name__, cb)

    def on_train_begin(self, epochs):
        for cb in self.cbs:
            cb.on_train_begin(epochs)

    def on_epoch_begin(self, epoch):
        self.learn.model.train()
        for cb in self.cbs:
            cb.on_epoch_begin(epoch)

    def on_epoch_end(self):
        for cb in self.cbs:
            cb.on_epoch_end()

    def on_batch_begin(self, batch):
        for cb in self.cbs:
            cb.on_batch_begin(batch)

    def on_batch_end(self):
        for cb in self.cbs:
            cb.on_batch_end()

    def on_loss_end(self, loss, out, yb):
        for cb in self.cbs:
            cb.on_loss_end(loss, out, yb)
        return self.learn.model.training

    def on_validate_begin(self):
        self.learn.model.eval()
        for cb in self.cbs:
            cb.on_validate_begin()

    def on_validate_end(self):
        for cb in self.cbs:
            cb.on_validate_end()


# class CudaCallback(Callback):
#     """Manages if data/ model is send to gpu or cpu

#     Args:
#         Callback (self): Implements alls methods
#     """
#     # def on_train_begin(self, epochs):
#     #     if self.learn.gpu:
#     #         # todo: error message if no gpu available
#     #         self.learn.model = self.learn.model.to(self.learn.device)

#     def on_batch_begin(self, batch):
#         self.xb, self.yb = batch[0], batch[1]
#         # todo: uncomment when CNN
#         self.xb = self.xb.unsqueeze(1)
#         if self.learn.gpu:
#             self.xb = self.xb.to(self.learn.device)
#             self.yb = self.yb.to(self.learn.device)
#         self.batch = (self.xb, self.yb)


class Recorder(Callback):
    """Tracking of  train/valid loss and setattr[yb, out, loss] to be available in all child classes

    Args:
        Callback (self): Implements alls methods
    """

    # todo: set self.history_raw on epoch end, in case that Monitor is not included
    #  * using numpy arrays for summming vals is much faster than lists

    def __init__(self, learn):
        super().__init__(learn)
        self.learn = learn
        self.history = {"epochs": [0]}
        self.best_values = {}
        self.new_best_values = {}
        self.batch_vals = {
            "train_loss": [],
            "valid_loss": [],
            "train_pred": [],
            "valid_pred": [],
        }
        self.monitor = []

        setattr(self.learn, "train_time", 0)

        if self.learn.resume:
            (
                self.learn.model,
                self.learn.opt,
                self.history,
                self.best_values,
                self.learn.train_time,
            ) = load_checkpoint(self.learn.resume, self.learn.model, self.learn.opt)

    def on_train_begin(self, epochs):
        self._starttime = time() - self.learn.train_time

        self.learn.model.train()
        self.epochs = epochs
        if self.monitor == []:
            self.monitor = ["train_loss", "valid_loss"]
        if not self.learn.resume:
            for mon in self.monitor:
                self.history[mon] = []

            for mon in self.monitor:
                if "loss" in mon:
                    self.best_values[mon] = [np.less, np.inf, np.inf]
                    self.history[mon].append(np.inf)
                if "acc" in mon:
                    self.best_values[mon] = [np.greater, -np.inf, -np.inf]
                    self.history[mon].append(-np.inf)
                self.new_best_values[mon] = False
        else:
            for mon in self.monitor:
                self.new_best_values[mon] = False
        setattr(self.learn, "best_values", self.best_values)
        setattr(self.learn, "history_raw", self.history)
        setattr(self.learn, "new_best_values", self.new_best_values)

    def on_epoch_begin(self, epoch):
        self.epoch = epoch
        self.batch_vals = {
            "train_loss": [],
            "valid_loss": [],
            "train_pred": [],
            "valid_pred": [],
        }

        for mon in self.monitor:
            self.learn.new_best_values[mon] = False

    def on_loss_end(self, loss, out, yb):
        self.yb = yb
        self.loss = loss
        self.out = out
        if self.learn.model.training:
            self.batch_vals["train_loss"].append(loss.item())
        else:
            self.batch_vals["valid_loss"].append(loss.item())

    def on_batch_end(self):
        _, batch_pred = torch.max(self.out.data, 1)
        batch_correct = (batch_pred == self.yb).sum().item() / len(self.yb)
        if self.learn.model.training:
            self.batch_vals["train_pred"].append(batch_correct)
        else:
            self.batch_vals["valid_pred"].append(batch_correct)

    def on_epoch_end(self):
        for mon in self.monitor:
            self.history[mon].append(getattr(self, mon)())
        self.history["epochs"].append(int(self.epoch + 1))

        self.learn.train_time = time() - self._starttime
        if self.verbose == True:
            self._print_console()

        self.learn.history_raw = self.history

        for mon in self.monitor:
            old_val = self.best_values[mon][2]
            new_val = self.history[mon][-1]
            comp = self.best_values[mon][0]
            if comp(new_val, old_val):
                self.best_values[mon][1] = self.best_values[mon][2]
                self.best_values[mon][2] = new_val
                self.learn.new_best_values[mon] = True

    def valid_acc(self):
        return sum(self.batch_vals["valid_pred"]) / len(self.batch_vals["valid_pred"])

    def valid_loss(self):
        return sum(self.batch_vals["valid_loss"]) / len(self.batch_vals["valid_loss"])

    def train_loss(self):
        return sum(self.batch_vals["train_loss"]) / len(self.batch_vals["train_loss"])

    def _print_console(self):

        out_string = f""
        out_string += f"epoch: {int(self.epoch)+1}/{self.epochs}\t["
        for key, val in self.history.items():
            if key != "epochs":
                out_string += f"{key}: {val[-1]:.4f}\t"
        out_string += f"train_time: {self._sec_conv_str(self.learn.train_time)}\t"
        print(out_string[:-1] + "]")

    def _sec_conv_str(self, time):
        t = time / 3600
        hours = str(int(t))
        minutes = str(int(t * 60 % 60))
        seconds = str(int(t * 3600 % 60))
        if len(hours) == 1:
            hours = "0" + hours
        if len(minutes) == 1:
            minutes = "0" + minutes
        if len(seconds) == 1:
            seconds = "0" + seconds

        return f"{hours}:{minutes}:{seconds}"


class EarlyStopping(Callback):
    def __init__(self, learn):
        super().__init__(learn)
        self.monitor = "valid_loss"
        self.patience = 20
        self.counter = 0
        self.min_delta = 1e-3

    def on_epoch_end(self):
        if self.learn.new_best_values[self.monitor]:
            diff = np.abs(
                self.learn.best_values[self.monitor][1]
                - self.learn.best_values[self.monitor][2]
            )
            if diff > self.min_delta:
                self.counter = 0
            else:
                self.counter += 1
        else:
            self.counter += 1

        if self.counter == self.patience:
            self.learn.do_stop = True


# plot func: option to plot specific path


class Checkpoints(Callback):
    # todo: finish docstring
    """Saves history and Pytorch models during training

    Args:
        TrackValues: Parent class tracks best values for train/ valid loss/acc

    Attr:
        VarAttr:
            monitor(str): quantity to be monitored
            history_format(fileformat): fileformat for panda.DataFrame.to_*'s method. [parquet for high compression and fast reading]
            delta(float): min change in monitored quantity to qualify as improvement
            ckp_path(str): path to save checkpoints
            no_time_path(str): if not specified in toml the current date is used as checkpoint foldername
            use_last_run(bool): if true, no new run directory is created. all files are saved in last run directory
            detailed_name(bool): if true, file names are saved with arch/bs/monitor information
            debug_timestamp(bool): if true, each run is saved in the last run folder, but with timeformat h/m/s
            resume(str): if specified in toml: history and model is loaded, before resuming training
        FuncAttr:
            on_train_begin: creates folder struct for new run and initiates function for comparison of best and new value
            on_epoch_begin: checks for new best value -> saves history and model statedict
            create_checkpoint_path: creates checkpoint directory structure

    Deps:
        Modules: on_train_begin -> datetime/ pytz/ np
        Functions: on_train_begin -> self.create_checkpoint_path
    """

    def on_train_begin(self, epochs):
        self.save_path = self.create_checkpoint_path()

        if self.detailed_name:
            self.save_name = (
                f"_Arch-{self.learn.arch}_bs-{self.learn.bs}_{self.monitor}"
            )
        else:
            self.save_name = f""

        if type(self.debug_timestamp) is bool:
            if self.debug_timestamp:
                timezone = pytz.timezone("Europe/Berlin")
                time = datetime.now()
                time = timezone.localize(time).strftime("%Y-%m-%dT%H_%M_%ST%z")
                self.save_name += f"_{time}"
        else:
            self.save_name += f"_{self.debug_timestamp}"

    def on_epoch_end(self):
        if self.save_model:
            checkpoint = {
                "best_values": self.learn.best_values,
                "history_raw": self.learn.history_raw,
                "state_dict": self.learn.model.state_dict(),
                "optimizer": self.learn.opt.state_dict(),
                "train_time": self.learn.train_time,
            }
            save_checkpoint(
                checkpoint, False, Path(self.save_path / f"model_{self.save_name}")
            )

            if self.learn.new_best_values[self.monitor]:
                diff = np.abs(
                    self.learn.best_values[self.monitor][1]
                    - self.learn.best_values[self.monitor][2]
                )
                if diff > self.min_delta:
                    # ? is this correct? or comp between best new value and old value
                    print(
                        f"best checkpoint: {self.monitor}: {self.learn.best_values[self.monitor][2]}"
                    )
                    save_checkpoint(
                        checkpoint,
                        True,
                        Path(self.save_path / f"model_{self.save_name}"),
                    )

        if self.save_history:
            df = pd.DataFrame(self.learn.history_raw).set_index("epochs")
            to_func = getattr(df.iloc[1:], "to_" + self.history_format)
            to_func(
                Path(self.save_path / f"history{self.save_name}.{self.history_format}")
            )

    def create_checkpoint_path(self):
        """Creating Checkpoint directory structure according to attributs set in setup.toml

        Returns:
            run_path(str): path to created checkpoint directory
        """

        if hasattr(self, "no_time_path"):
            datetime_now = self.no_time_path
        else:
            datetime_now = datetime.now().strftime("%Y-%m-%d")

        if self.use_last_run is not True:
            curr_path = Path(self.ckp_path + "/" + datetime_now)
        else:
            curr_path = Path(self.ckp_path)
            subdirs = []
            for path in curr_path.iterdir():
                if path.is_dir():
                    subdirs.append(datetime.fromisoformat(path.name))
            last_run_date = max(subdirs).strftime("%Y-%m-%d")
            curr_path = Path(curr_path / last_run_date)

        curr_path.mkdir(parents=True, exist_ok=True)

        run_dirs = []
        for path in curr_path.iterdir():
            if path.is_dir():
                run_dirs.append(path.name)

        if self.use_last_run and run_dirs:
            run_path = curr_path / run_dirs[-1]
        elif not self.use_last_run and run_dirs:
            run_num = int(run_dirs[-1][-3:]) + 1
            for i in range(3 - len(str(run_num))):
                run_num = "0" + str(run_num)
            run_path = curr_path / Path("run_" + run_num)
            run_path.mkdir(parents=True, exist_ok=True)
        # * not sure if this is needed: (use_last_run and not run_dirs) or (not use_last_run and not run_dirs):
        else:
            run_path = curr_path / Path("run_001")
            run_path.mkdir(parents=True, exist_ok=True)
        return run_path


class Tensorboard(Callback):
    def __init__(self, learn):
        super().__init__(learn)
    
    def on_train_begin(self, epochs):
        self.writer = SummaryWriter(log_dir=self.logdir)
        img, _ = next(iter(self.learn.data.train_dl))   
        
        img = img.unsqueeze(1).to(self.learn.device)
        self.writer.add_graph(self.learn.model.to(self.learn.device), img)
        
    def on_epoch_end(self):
        hist = self.learn.history_raw
        
        for key, val in hist.items():
            epoch = hist["epochs"][-1]
            if key != "epochs":
                if "loss" in key:
                    self.writer.add_scalar("loss/" + key, val[-1], epoch)
                if "acc" in key:
                    self.writer.add_scalar("acc/" + key, val[-1], epoch)
                    
        
def save_checkpoint(state, is_best, checkpoint_path):
    """save best and last pytorch model in checkpoint_path

    Args:
        state(dict): checkpoint to save
        is_best(bool): getting bool from metric function
        checkpoint_path(str): path to save checkpoint
    """
    save_path = checkpoint_path.as_posix() + ".pt"
    torch.save(state, save_path)
    if is_best:
        best_path = checkpoint_path.as_posix() + "_best.pt"
        shutil.copyfile(save_path, best_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    """loads pytorch model and history

    Args:
        checkpoint_path(str): path to save checkpoint
        model(nn.Model): model to load checkpoint parameters into
        optimizer(torch.optim): in previous training defined optimizer
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    history = checkpoint["history_raw"]
    best_history = checkpoint["best_values"]
    resume_time = checkpoint["train_time"]
    return model, optimizer, history, best_history, resume_time


def get_callbacks(setup_config, learn):
    """Loading Callback classes according to setup.toml

    Args:
        setup_config (dict): contains information of callbacks and attributes

    Returns:
        list: returns list of all callback classes
    """
    implemented_cbs = {"r": "Recorder", "e": "EarlyStopping", "c": "Checkpoints", "t": "Tensorboard"}

    cb_list = [c for c in setup_config if c[:2] == "c_"]

    cb_args = {}
    for i in cb_list:
        cb = i.split("_", 2)[:]
        if cb[1] not in cb_args:
            cb_args[cb[1]] = {cb[2]: setup_config[i]}
        else:
            cb_args[cb[1]][cb[2]] = setup_config[i]
    cbs = []
    for _cb, cb_list in cb_args.items():
        # * important, that classes get instantiated here
        if implemented_cbs[_cb] in globals():
            cb = globals()[implemented_cbs[_cb]](learn)
            for attr, val in cb_list.items():
                setattr(cb, attr, val)
            cbs.append(cb)
    return cbs


def get_callbackhandler(setup_config, learn):
    """Creates Callbackhandler with Callbacks from setup.toml

    Args:
        setup_config (dict): contains information of callbacks and attributes
    Deps:
        get_callbacks(dict): loading and returning callback classes according to setup.toml as list
    Returns:
        CallbackHandler: returns CallbackHandler class with all callbacks

    Adds:
        Recorder and CudaCallback are added automatically in case of no Callbacks in setup.toml
    """
    # if any([c for c in setup_config.keys() if c[:2] == "c_"]):
    #     cbs = [CudaCallback(learn)]
    #     cbs.extend(get_callbacks(setup_config, learn))
    # else:
    #     cbs = [CudaCallback(learn)]
    return CallbackHandler(get_callbacks(setup_config, learn), learn)
