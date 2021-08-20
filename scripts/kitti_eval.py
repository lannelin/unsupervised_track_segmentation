import os
from typing import Dict, Tuple

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from pl_bolts.datamodules.kitti_datamodule import KittiDataModule
from pytorch_lightning import Trainer, seed_everything
from tqdm.auto import tqdm

# isort:imports-firstparty
from trackseg.datamodules import SingleImageDataModule
from trackseg.model import UnsupervisedSemSegment

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))


def get_results_fname(
    n_channels: int, max_steps: int, resize_size: Tuple[int, int]
) -> str:
    return (
        f"kitti_results_{n_channels}channels_{max_steps}steps"
        f"_halfsize_resize{resize_size[0]}x{resize_size[1]}.csv"
    )


def train_validate(
    image: torch.Tensor,
    target: torch.Tensor,
    n_channels: int,
    max_steps: int,
    resize_size: Tuple[int, int],
) -> Dict[str, float]:
    # don't need target for inner dm
    inner_dm = SingleImageDataModule(
        image=image,
        target=target,
        im_size=resize_size,
        normalize=False,
        num_workers=0,
    )
    # new model
    model = UnsupervisedSemSegment(
        n_channels=n_channels, connectivity_weight=1.0, similarity_weight=1.0
    )
    # train
    trainer = Trainer(
        gpus=1,
        max_steps=max_steps,
        #        weights_save_path=save_dir,
    )
    trainer.fit(model, datamodule=inner_dm)
    model.eval()
    eval_dl = inner_dm.train_dataloader()
    results = trainer.validate(dataloaders=eval_dl)
    # only expect one result so take 0th index
    results = results[0]
    return results


def main(
    kitti_data_path: str,
    kitti_val_split: float,
    kitti_test_split: float,
    n_channels: int,
    max_steps: int,
    resize_size: Tuple[int, int],
    result_dir: str,
    random_seed: int,
):
    seed_everything(random_seed)

    dm = KittiDataModule(
        kitti_data_path,
        batch_size=1,
        val_split=kitti_val_split,
        test_split=kitti_test_split,
        seed=random_seed,
        num_workers=0,
    )

    all_results = list()
    # loop through val dataloader
    for batch in tqdm(dm.val_dataloader(), desc="kitti val dataloader - outer loop"):
        images, targets = batch
        all_results.append(
            train_validate(
                image=images[0],
                target=targets[0],
                n_channels=n_channels,
                max_steps=max_steps,
                resize_size=resize_size,
            )
        )

    df = pd.DataFrame(all_results)
    result_fpath = os.path.join(
        result_dir,
        get_results_fname(
            n_channels=n_channels, max_steps=max_steps, resize_size=resize_size
        ),
    )
    df.to_csv(result_fpath)


@hydra.main(config_path="../config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    main(
        kitti_data_path=os.path.join(PROJECT_ROOT, cfg.locations.data.kitti),
        kitti_val_split=cfg.kitti.val_split,
        kitti_test_split=cfg.kitti.test_split,
        n_channels=cfg.model.n_channels,
        max_steps=cfg.model.max_steps,
        resize_size=tuple(cfg.kitti.resize_size),
        result_dir=os.path.join(PROJECT_ROOT, cfg.locations.results),
        random_seed=cfg.general.random_seed,
    )


if __name__ == "__main__":
    my_app()
