from pathlib import Path

import click
import h5py
import numpy as np
import toml
from core.dl_framework.data import get_transforms
from core.dl_framework.utils import read_config
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


@click.command()
@click.argument(
    "config_file_path", default="configs/default_train_config.toml", type=Path
)
@click.option(
    "--img_data_path", "-p", default="/seg_train/seg_train", type=str
)
@click.option("--name", "-n", default="processed_train", type=str)
@click.option("--compress", "-c", default=False, type=bool)
def main(
    config_file_path: Path, img_data_path: str, name: str, compress: bool
) -> None:
    """Script that stores all imgs files as numpy array and saves it as h5 file.

    Args:
        config_file_path (Path): path to config file
        img_data_path (str): path to img files. all files of each class has to be stored in a seperate folder
        name (str): name that is used as key for h5 file. data can be accessed via: <name>_x, <name>_y
        compress (bool): option to compress dataset with gzip
    """
    config_file = read_config(toml.load(config_file_path))

    data_path = Path(config_file["p_tmp_data_path"]) / img_data_path[1:]
    save_path = Path(config_file["p_local_data_path"] + "/processed_files")
    save_path.mkdir(parents=True, exist_ok=True)

    transform = get_transforms(config_file)
    for trans in transform:
        if type(trans).__name__ == "Resize":
            all_imgs = [x for x in data_path.rglob("*.jpg") if x.is_file()]

            y_classes = [y.name for y in Path(data_path).glob("*")]
            classes_to_label = dict(zip(y_classes, range(len(y_classes))))

            resized_imgs = []
            labels = []
            pbar_imgs_list = tqdm(all_imgs, total=len(all_imgs), leave=True)

            for img in pbar_imgs_list:
                labels.append(classes_to_label[img.parent.name])
                resized_imgs.append(trans(Image.open(img)))

            resized_imgs_arr = np.array(
                [np.array(img) for img in resized_imgs]
            )
            labels_arr = np.array(labels)

            with h5py.File(
                Path(save_path / name).with_suffix(".h5"), "w", libver="latest"
            ) as f:
                key_name = name.split("_")[-1]
                if compress:
                    f.create_dataset(
                        f"{key_name}_x",
                        data=resized_imgs_arr,
                        compression="gzip",
                        compression_opts=9,
                        chunks=True
                    )
                    f.create_dataset(
                        f"{key_name}_y",
                        data=labels_arr,
                        compression="gzip",
                        compression_opts=9,
                        chunks=True
                    )
                else:
                    f.create_dataset(f"{key_name}_x", data=resized_imgs_arr, chunks=True)
                    f.create_dataset(f"{key_name}_y", data=labels_arr, chunks=True)


if __name__ == "__main__":
    main()
