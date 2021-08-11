import os
from typing import Callable, Tuple

import numpy as np
import torch
from PIL import Image
from pl_bolts.datamodules.kitti_datamodule import KittiDataModule
from pytorch_lightning import Trainer, seed_everything
from torchvision import transforms as transforms_lib

# isort:imports-firstparty
from trackseg.datamodules import SingleImageDataModule
from trackseg.model import UnsupervisedSemSegment

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))

# TODO move to config file? or to args?
SEED = 42
TEST_SPLIT = 0.1  # 200 images total: 0.1 = 20
N_CLUST = 50
SAVE_DIR = os.path.join(PROJECT_ROOT, "lightning_out", "kitt_eval_runs")
RESIZE_SIZE = (188, 621)  # orig kitti size = (376, 1242)
KITTI_DATA = os.path.join(PROJECT_ROOT, "data", "data_semantics")

seed_everything(SEED)


# need a separate transformation that *doesn't* normalize as already done
def transform_w_resize(resize_size: Tuple[int, int]) -> Callable:
    kitti_transforms = transforms_lib.Compose(
        [
            transforms_lib.Resize(size=resize_size),
            transforms_lib.ToTensor(),
        ]
    )
    return kitti_transforms


def binary_iou(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape
    x = a + b
    iou = (x == 2).sum() / (x > 0).sum()

    return iou


transforms = transform_w_resize(resize_size=RESIZE_SIZE)

dm = KittiDataModule(
    KITTI_DATA,
    batch_size=1,
    test_split=TEST_SPLIT,
    seed=SEED,
    num_workers=0,
)
dl = dm.test_dataloader()

assert dl.batch_size == 1

for i, (images, targets) in enumerate(dl):
    image = images[0]
    target = targets[0]
    save_dir = os.path.join(SAVE_DIR, str(i))
    # don't need target for inner dm
    inner_dm = SingleImageDataModule(image=image, train_transforms=transforms)
    # model
    model = UnsupervisedSemSegment(
        n_clusters=N_CLUST, connectivity_weight=1.0, similarity_weight=1.0
    )
    # train
    trainer = Trainer(
        gpus=1,
        max_steps=200,
        weights_save_path=save_dir,
    )
    trainer.fit(model, datamodule=inner_dm)

    model.eval()

    out = model(images)[0].detach().cpu()

    pred = out.argmax(dim=0)

    assert pred.max() < 256  # for resizing (need to convert to im) purposes

    # MUST USE NEAREST RESAMPLING TO PREVENT NEW CLASSES BEING INTRODUCED
    resized_pred_im = Image.fromarray(pred.numpy().astype(np.uint8)).resize(
        (target.shape[1], target.shape[0]),
        resample=Image.NEAREST,
    )
    resized_pred = np.asarray(resized_pred_im)

    pred_classes = np.unique(resized_pred)
    true_classes = np.unique(target)
    len(pred_classes), len(true_classes)

    all_ious = torch.zeros((len(pred_classes), len(true_classes)))
    for i, pred_class in enumerate(pred_classes):
        pred_class_ious = dict()
        a = torch.from_numpy((resized_pred == pred_class).astype(int))
        for j, true_class in enumerate(true_classes):
            b = (target == true_class).int()
            all_ious[i, j] = binary_iou(a, b)

    # TODO review - allows single assigned cluster to match many truth clusters
    # review-cont: is this possible to game accidently?
    # take max along prediction dim - getting best match for each truth cluster
    score_a = all_ious.max(dim=0).values.mean()
    # max along truth dim - getting best match for each prediction cluster
    # (penalises small prediction clusters?)
    score_b = all_ious.max(dim=1).values.mean()

    print(score_a.item(), score_b.item())
    raise NotImplementedError("not finished yet!")
