import os

from pl_bolts.datamodules.kitti_datamodule import KittiDataModule
from pytorch_lightning import Trainer, seed_everything

# isort:imports-firstparty
from trackseg.datamodules import SingleImageDataModule
from trackseg.model import UnsupervisedSemSegment

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))

# TODO move to config file? or to args?
# TODO save dir only controlling checkpoints, not logs & hparams
SEED = 42
TEST_SPLIT = 0.1  # 200 images total: 0.1 = 20
N_CHANNELS = 50
SAVE_DIR = os.path.join(PROJECT_ROOT, "lightning_out", "kitt_eval_runs")
RESIZE_SIZE = (188, 621)  # orig kitti size = (376, 1242)
KITTI_DATA = os.path.join(PROJECT_ROOT, "data", "data_semantics")
MAX_STEPS = 10

seed_everything(SEED)

dm = KittiDataModule(
    KITTI_DATA,
    batch_size=1,
    test_split=TEST_SPLIT,
    seed=SEED,
    num_workers=0,
)
dl = dm.val_dataloader()

assert dl.batch_size == 1

for i, (images, targets) in enumerate(dl):
    image = images[0]
    target = targets[0]
    save_dir = os.path.join(SAVE_DIR, str(i))
    # don't need target for inner dm
    inner_dm = SingleImageDataModule(image=image,
                                     target=target,
                                     normalize=False,
                                     num_workers=0)
    # model
    model = UnsupervisedSemSegment(
        n_channels=N_CHANNELS, connectivity_weight=1.0, similarity_weight=1.0
    )
    # train
    trainer = Trainer(
        gpus=1,
        max_steps=MAX_STEPS,
        weights_save_path=save_dir,
    )
    trainer.fit(model, datamodule=inner_dm)

    model.eval()
    # TODO resize images before val?
    eval_dl = inner_dm.train_dataloader()
    trainer.validate(dataloaders=eval_dl)

    raise NotImplementedError("not finished yet!")
