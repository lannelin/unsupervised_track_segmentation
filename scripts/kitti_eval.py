import os
from typing import Dict

import pandas as pd
import torch
from pl_bolts.datamodules.kitti_datamodule import KittiDataModule
from pytorch_lightning import Trainer, seed_everything
from tqdm.auto import tqdm

# isort:imports-firstparty
from trackseg.datamodules import SingleImageDataModule
from trackseg.model import UnsupervisedSemSegment

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))

# TODO move to config file? or to args?
# TODO save dir only controlling checkpoints, not logs & hparams
SEED = 42
VAL_SPLIT = 0.1  # 200 images total: 0.1 = 20
TEST_SPLIT = 0.1  # 200 images total: 0.1 = 20
N_CHANNELS = 48
SAVE_DIR = os.path.join(PROJECT_ROOT, "lightning_out", "kitt_eval_runs")
RESIZE_SIZE = (188, 621)  # orig kitti size = (376, 1242)
KITTI_DATA = os.path.join(PROJECT_ROOT, "data", "data_semantics")
MAX_STEPS = 200
RESULT_FPATH = f"./kitti_results_{N_CHANNELS}channels_{MAX_STEPS}steps_halfsize.csv"

seed_everything(SEED)


def train_validate(image: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    # don't need target for inner dm
    inner_dm = SingleImageDataModule(
        image=image, target=target, normalize=False, num_workers=0
    )
    # new model
    model = UnsupervisedSemSegment(
        n_channels=N_CHANNELS, connectivity_weight=1.0, similarity_weight=1.0
    )
    # train
    trainer = Trainer(
        gpus=1,
        max_steps=MAX_STEPS,
        #        weights_save_path=save_dir,
    )
    trainer.fit(model, datamodule=inner_dm)
    model.eval()
    # TODO resize images before val?
    eval_dl = inner_dm.train_dataloader()
    results = trainer.validate(dataloaders=eval_dl)
    # only expect one result so take 0th index
    results = results[0]
    return results


def main():
    dm = KittiDataModule(
        KITTI_DATA,
        batch_size=1,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        seed=SEED,
        num_workers=0,
    )

    all_results = list()
    # loop through val dataloader
    for batch in tqdm(dm.val_dataloader(), desc="kitti val dataloader - outer loop"):
        images, targets = batch
        all_results.append(train_validate(image=images[0], target=targets[0]))

    df = pd.DataFrame(all_results)
    df.to_csv(path=RESULT_FPATH)


if __name__ == "__main__":
    main()
