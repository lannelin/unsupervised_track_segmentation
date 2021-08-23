import os
from typing import Dict, Tuple

import hydra
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig
from pl_bolts.datamodules.kitti_datamodule import KittiDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from tqdm.auto import tqdm

# isort:imports-firstparty
from trackseg.datamodules import UnsupervisedSingleImageDataModule
from trackseg.model import UnsupervisedSemSegment

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))


def train_validate(
    image: torch.Tensor,
    target: torch.Tensor,
    n_channels: int,
    max_steps: int,
    resize_size: Tuple[int, int],
    run_id: str,
    iteration: int,
) -> Dict[str, float]:
    # don't need target for inner dm
    inner_dm = UnsupervisedSingleImageDataModule(
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
    wandb_logger = WandbLogger(id=run_id, prefix=str(iteration))
    trainer = Trainer(gpus=1, max_steps=max_steps, logger=wandb_logger)

    trainer.fit(model, datamodule=inner_dm)
    model.eval()
    # use test_dataloader here that also returns target
    results = trainer.validate(dataloaders=inner_dm.test_dataloader())

    # close logger
    trainer.logger.close()

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
    wandb_project_name: str,
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

    wandb_run = wandb.init(project=wandb_project_name)

    all_results = list()
    # loop through val dataloader
    for i, batch in enumerate(
        tqdm(dm.val_dataloader(), desc="kitti val dataloader - outer loop")
    ):
        images, targets = batch
        all_results.append(
            train_validate(
                image=images[0],
                target=targets[0],
                n_channels=n_channels,
                max_steps=max_steps,
                resize_size=resize_size,
                run_id=wandb_run.id,
                iteration=i,
            )
        )

    df = pd.DataFrame(all_results)
    result_dict = df.mean().add_prefix("FINAL_").to_dict()
    wandb_run.log(result_dict)
    wandb_run.finish()


@hydra.main(config_path="../config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    main(
        kitti_data_path=os.path.join(PROJECT_ROOT, cfg.locations.data.kitti),
        kitti_val_split=cfg.kitti.val_split,
        kitti_test_split=cfg.kitti.test_split,
        n_channels=cfg.model.n_channels,
        max_steps=cfg.model.max_steps,
        resize_size=tuple(cfg.kitti.resize_size),
        wandb_project_name=cfg.wandb.project,
        random_seed=cfg.general.random_seed,
    )


if __name__ == "__main__":
    my_app()
