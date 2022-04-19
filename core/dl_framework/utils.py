import itertools
import os
from datetime import datetime
from pathlib import Path

import plotly.io as pio

pio.renderers.default = "notebook_connected"
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
from plotly.subplots import make_subplots


def read_config(config_file):
    setup_config = {}
    for key, value in config_file.items():
        if key == "title":
            continue
        else:
            for subkey, subval in value.items():
                if type(subval) != dict:
                    key_name = f"{key[0]}_{subkey}"
                    setup_config[key_name] = subval
                else:
                    for subsubkey, subsubval in subval.items():
                        key_name = f"{key[0]}_{subkey[0]}_{subsubkey}"
                        setup_config[key_name] = subsubval
    return setup_config


def get_history(ckp_path, monitor, fileformat=["csv"]):
    project_name = Path(os.getcwd()).name
    for i, path in enumerate(ckp_path.parts):
        length = len(ckp_path.parts)
        if (path == project_name) and (i+1 != length):
            project_path = ckp_path.parents[length-i-2]

    files = []
    if type(fileformat) != list:
        fileformat = [fileformat]
    for fmt in fileformat:
        for rel_path in ckp_path.rglob("*."+fmt):
            # files.append(rel_path.relative_to(project_path))
            files.append(rel_path)
        metric_files = []
        for metric in files:
            if monitor in metric.name:
                metric_files.append(metric)

    return metric_files


def get_specific_history(ckp_path, monitor, fileformat=["csv"], specific="best"):
    # todo: add if statement to load specific date/run history
    files = get_history(ckp_path, monitor, fileformat)
    if specific == "best":
        best_vals = []
        for idx, hist in enumerate(files):
            get_func = getattr(pd, "read_" + hist.suffix[1:])
            tmp_df = get_func(hist)
            if "acc" in monitor:
                best_vals.append(tmp_df[monitor].max())
                best_idx = best_vals.index(max(best_vals))
            if "loss" in monitor:
                best_vals.append(tmp_df[monitor].min())
                best_idx = best_vals.index(min(best_vals))
        return [[getattr(pd, "read_" + files[best_idx].suffix[1:])(files[best_idx])], files[best_idx]]
    elif specific == "all":
        df_list = []
        for idx, hist in enumerate(files):
            get_func = getattr(pd, "read_" + hist.suffix[1:])
            tmp_df = get_func(hist)
            df_list.append(tmp_df)
        return [df_list, "all"]
    else:
        pass

    
def plot_history(history):
    # todo: add functionality to get list of history_files and plot each file or plot all files in one
    col_pal = px.colors.sequential.Rainbow
    col_pal_iter = itertools.cycle(col_pal) 
    implemented_metric = ["acc", "loss"]
    plot_dict = {}
    for met in implemented_metric:
        plot_dict[met] = [s for s in history if met in s]

    fig = make_subplots(rows=1, cols=len(plot_dict), subplot_titles=list(plot_dict.keys()))

    for idx, (met, mets) in enumerate(plot_dict.items()):
        for key in mets:
            new_color = next(col_pal_iter)
            new_color = next(col_pal_iter)
            fig.add_trace(go.Scatter(x=history.index, y=history[key].values, name=key, line=dict(color=new_color)), row=1, col=idx+1)
            dmin, dmax = history[key].min(), history[key].max()
            if met == "acc":
                epoch = history[key].argmax()+1
                fig.add_trace(go.Scatter(x=[epoch], y=[dmax], name="best val", line=dict(color=new_color, width=1, dash="dash")), row=1, col=idx+1)
                fig.add_vline(x=epoch, line_width=1, line_color=new_color, line_dash="dash", row=1, col=idx+1)
            if met == "loss":
                epoch = history[key].argmin()+1
                fig.add_trace(go.Scatter(x=[epoch], y=[dmin], name="best val", line=dict(color=new_color, width=1, dash="dash")), row=1, col=idx+1)
                fig.add_vline(x=epoch, line_width=1, line_color=new_color, line_dash="dash", row=1, col=idx+1)
    # learn.history.keys()[0]
    # fig.update_layout(title_text="test")
    fig.show()
    
def plot_history_all(history_list):
    history, add_info = history_list[0], history_list[1]
    for hist in history:
        plot_history(hist)
