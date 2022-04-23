from pathlib import Path

import click
import numpy as np
import toml
from core.dl_framework.data import get_transforms
from core.dl_framework.utils import read_config
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

@click.command()
@click.argument("config_file_path", default="configs/default_train_config.toml", type=Path)
@click.argument("train_data_path", default="/seg_train/seg_train", type=str)
@click.argument("name", default="processed_train", type=str)

def main(config_file_path, train_data_path, name):
    config_file = read_config(toml.load(config_file_path))
    data_path = Path(config_file["p_tmp_data_path"] + "/" + train_data_path)
    
    save_path = Path(config_file["p_local_data_path"] + "/processed_files")
    save_path.mkdir(parents=True, exist_ok=True)
    
    transform = get_transforms(config_file)
    for trans in transform:
        if type(trans).__name__ == "Resize":
            all_imgs = [x for x in data_path.rglob("*.jpg") if x.is_file()]
            resized_imgs = []
            pbar_imgs_list = tqdm(all_imgs, total=len(all_imgs), leave=True)
            for img in pbar_imgs_list:
                resized_imgs.append(trans(Image.open(img)))
            np.savez_compressed(save_path/name, np.array(resized_imgs))

if __name__ == "__main__":
    main()
