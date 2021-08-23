from typing import Any, Callable, Optional, Tuple

# isort:imports-thirdparty
import torch
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

# isort:imports-firstparty
from trackseg.datasets import LocalMP4FramesDataset, SingleImageDataset

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
else:  # pragma: no cover
    warn_missing_pkg("torchvision")

IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDS = [0.229, 0.224, 0.225]


def get_transforms(
    normalize: Optional[bool],
    resize_size: Optional[Tuple[int, int]],
) -> Callable:
    transforms = list()
    if resize_size is not None:
        transforms.append(transform_lib.Resize(size=resize_size))
    if normalize is not None and normalize:
        transforms.append(
            transform_lib.Normalize(mean=IMAGENET_MEANS, std=IMAGENET_STDS)
        )
    transforms.append(transform_lib.ToTensor())
    transforms = transform_lib.Compose(transforms)

    return transforms


class UnsupervisedSegmentationDataModule(LightningDataModule):
    """


    Unsupervised Segmentation data and transforms

    Transforms::

        video_mean = ...
        video_std = ...
        transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            transform_lib.Normalize(
                mean=video_mean,
                std=video_std
            )
        ])

    example:
    ```
        dm = UnsupervisedSegmentationDataModule(
            name="example_dataZ",
            mp4_fpath="/path/to/video.mp4",
            data_dir="/path/to/data",
            im_size=(256, 480),
            desired_frame_rate=20,
            video_start_sec=1,
            video_end_sec=3,
            batch_size=4,
        )
        dl = dm.train_dataloader()
        for batch in dl:
            im = batch[0].permute(1, 2, 0)  # HWC
    ```
    """

    name = "UnsupervisedSegmentation"
    extra_args: dict = {}

    def __init__(
        self,
        name: str,
        mp4_fpath: str,
        data_dir: str,
        im_size: Optional[Tuple[int, int]],
        desired_frame_rate: float,
        allow_above_fps: bool = False,
        video_start_sec: int = 0,
        video_end_sec: Optional[int] = None,
        num_workers: int = 16,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            mp4_fpath: path to mp4. Can be None if extracted_data_dir already populated
            data_dir: path to save frames of extracted video
            num_workers: how many workers to use for loading data
            batch_size: number of examples per training/eval step
            seed: random seed to be used for train/val/test splits
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned
             memory before returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "Transforms require `torchvision` which is not installed yet."
            )

        self.name = name
        self.mp4_fpath = mp4_fpath
        self.data_dir = data_dir
        self.resize_size = im_size

        self.desired_frame_rate = desired_frame_rate
        self.allow_above_fps = allow_above_fps
        self.video_start_sec = video_start_sec
        self.video_end_sec = video_end_sec

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.target_transforms = None

    @property
    def num_classes(self) -> int:
        """
        Return:
            -1
        """
        # TODO
        return -1

    def train_dataloader(self) -> DataLoader:
        """
        train set
        """

        # TODO im size??

        transforms = self.train_transforms or get_transforms(
            normalize=True,
            resize_size=self.resize_size,
        )

        dataset = LocalMP4FramesDataset(
            name=self.name,
            mp4_fpath=self.mp4_fpath,
            data_dir=self.data_dir,
            desired_fps=self.desired_frame_rate,
            allow_above_fps=self.allow_above_fps,
            video_start_sec=self.video_start_sec,
            video_end_sec=self.video_end_sec,
            transform=transforms,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader


class UnsupervisedSingleImageDataModule(LightningDataModule):
    train_dataset = None
    test_dataset = None

    def __init__(
        self,
        image: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        im_size: Optional[Tuple[int, int]] = None,
        num_workers: int = 1,
        seed: int = 42,
        pin_memory: bool = False,
        normalize: Optional[bool] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.resize_size = im_size

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "Transforms require `torchvision` which is not installed yet."
            )

        self.num_workers = num_workers
        self.batch_size = 1
        self.seed = seed
        self.shuffle = False
        self.pin_memory = pin_memory
        self.drop_last = False

        image_transforms = self.train_transforms or get_transforms(
            normalize=normalize, resize_size=self.resize_size
        )

        target_transforms = get_transforms(
            normalize=False, resize_size=self.resize_size
        )

        self.train_dataset = SingleImageDataset(
            image=image,
            target=None,
            image_transform=image_transforms,
            target_transform=target_transforms,
        )

        if target is not None:
            self.test_dataset = SingleImageDataset(
                image=image,
                target=target,
                image_transform=image_transforms,
                target_transform=target_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
