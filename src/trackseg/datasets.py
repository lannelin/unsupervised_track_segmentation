import logging
import math
import os
from typing import Callable, Optional, Tuple

import torch
from pl_bolts.datasets import LightDataset
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
from torch import Tensor
from torch.utils.data import Dataset

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms as transform_lib
    from torchvision.io import read_video
else:  # pragma: no cover
    warn_missing_pkg("torchvision")

logger = logging.getLogger(__name__)


class LocalMP4FramesDataset(LightDataset):
    """
    MP4 Dataset - loads an mp4 and extracts frames for use in vision tasks

    Args:
        name: name of the dataset - used for saving
        mp4_fpath: Filepath of mp4 video
        data_dir: Directory to save extracted frames to
        desired_fps: frame rate (fps) to sample from video (cannot exceed video fps)
        allow_above_fps: whether to allow fps to be rounded up from desired_fps
        video_start_sec: start time for extracting frames from video
        video_end_sec: end time for extracting frames from video
        transform: transform to apply to data - overrides default

    TODO Examples:
    """

    cache_folder_name = "cache"
    DATASET_NAME = "LocalMP4Dataset"
    transform = None
    actual_fps = None

    def __init__(
        self,
        name: str,
        mp4_fpath: str,
        data_dir: str,
        desired_fps: Optional[float],
        allow_above_fps: bool,
        video_start_sec: int = 0,
        video_end_sec: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.dir_path = data_dir
        self.desired_fps = desired_fps
        self.video_start_sec = video_start_sec
        self.video_end_sec = video_end_sec
        self.allow_above_fps = allow_above_fps
        self.transform = transform

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "Loading mp4 and frames requires `torchvision` which is not installed."
            )

        start_str = "" if video_start_sec is None else f"_{video_start_sec}start"
        end_str = "" if video_end_sec is None else f"_{video_end_sec}start"
        train_file_name = f"{name}_{desired_fps}fps{start_str}{end_str}.pt"

        os.makedirs(self.cached_folder_path, exist_ok=True)
        data_file = os.path.join(self.cached_folder_path, train_file_name)
        self.prepare_data(mp4_fpath=mp4_fpath, cached_train_filepath=data_file)
        if not os.path.isfile(data_file):
            raise RuntimeError("Dataset not found.")

        self.data = torch.load(data_file)

    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        img = self.data[idx].permute((2, 0, 1))  # to CHW

        if self.transform is not None:
            to_pil = transform_lib.ToPILImage()
            img = to_pil(img)  # then to PIL
            img = self.transform(img)

        return img

    def prepare_data(self, mp4_fpath: Optional[str], cached_train_filepath: str):
        if os.path.exists(cached_train_filepath):
            logger.warning(f"using cached dataset at {cached_train_filepath}")
            return

        if mp4_fpath is None or not os.path.isfile(mp4_fpath):
            raise ValueError(f"Specified mp4 filepath not valid: {mp4_fpath}")

        self.extract_mp4(mp4_fpath=mp4_fpath, cached_filepath=cached_train_filepath)

    def extract_mp4(self, mp4_fpath: str, cached_filepath: str) -> None:
        """Extract the mp4 to the data folder, saving at given frame rate"""
        logger.warning("extracting video and caching frames...")
        frames, _, meta = read_video(
            mp4_fpath,
            start_pts=self.video_start_sec,
            end_pts=self.video_end_sec,
            pts_unit="sec",
        )
        assert frames.ndim == 4

        fps = meta["video_fps"]

        if self.desired_fps is None:
            actual_fps = fps
        elif self.desired_fps > fps:
            actual_fps = fps
            logger.warning(
                f"specified `desired_fps` ({self.desired_fps})"
                f" higher than video fps ({fps}). Using {fps}fps."
            )
        elif self.allow_above_fps:
            scalar = round(fps / self.desired_fps)
            actual_fps = fps / scalar
            logger.warning(
                f"resampling to nearest fraction of existing fps ({fps}): {actual_fps}"
            )
        else:
            scalar = math.ceil(fps / self.desired_fps)
            actual_fps = fps / scalar
            logger.warning(
                f"resampling to floor fraction of existing fps ({fps}): {actual_fps}"
            )

        self.actual_fps = actual_fps
        frame_step = fps / self.actual_fps
        assert frame_step % 1 == 0
        frame_step = int(frame_step)
        frames = frames[::frame_step]
        frames = frames / 255.0
        torch.save(frames, cached_filepath)
        logger.warning("extracting video and caching frames... done")


class SingleImageDataset(Dataset):
    def __init__(
        self,
        image: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        image_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.data = image.unsqueeze(0)
        self.target = target.unsqueeze(0) if target is not None else None
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = self.data[idx]  # CHW

        if self.image_transform is not None:
            to_pil = transform_lib.ToPILImage()
            img = to_pil(img)  # then to PIL
            img = self.image_transform(img)

        if self.target is None:
            # if no target then just return image
            return img
        else:
            # otherwise check for target transform and return image and target
            target = self.target[idx]
            if self.target_transform is not None:
                to_pil = transform_lib.ToPILImage()
                target = to_pil(target)  # then to PIL
                target = self.target_transform(target).squeeze() # drop first dim

            return img, target
