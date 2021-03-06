from argparse import ArgumentParser
from typing import Tuple

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F

# isort:imports-firstparty
from trackseg.utils import binary_iou


# CNN model
# taken from https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
class MyNet(nn.Module):
    def __init__(self, input_dim: int, n_channel: int = 100, n_conv: int = 2):
        super(MyNet, self).__init__()

        self.n_conv = n_conv
        self.n_channel = n_channel

        self.conv1 = nn.Conv2d(input_dim, n_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(n_conv - 1):
            self.conv2.append(
                nn.Conv2d(n_channel, n_channel, kernel_size=3, stride=1, padding=1)
            )
            self.bn2.append(nn.BatchNorm2d(n_channel))
        self.conv3 = nn.Conv2d(n_channel, n_channel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(n_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(self.n_conv - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


# based on https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/vision/segmentation.py
# TODO exit on sufficiently few labels (min_labels) - may prevent collapse?
class UnsupervisedSemSegment(LightningModule):
    def __init__(
        self,
        n_channels: int,
        similarity_weight: float = 1.0,
        connectivity_weight: float = 1.0,
        lr: float = 0.1,
    ):
        """
        Args:
            n_channels: no. of channels to allow for in output
            similarity_weight: weight for similarity loss component (default 1.0)
            connectivity_weight: weight for connectivity loss component (defafult 1.0)
            lr: learning (default 0.1)
        """
        super().__init__()

        self.save_hyperparameters()

        self.similarity_weight = similarity_weight
        self.connectivity_weight = connectivity_weight
        self.lr = lr
        self.n_channels = n_channels

        self.net = MyNet(input_dim=3, n_channel=n_channels, n_conv=2)

        self.similarity_loss_fn = nn.CrossEntropyLoss()
        self.connectivity_loss_fn = nn.L1Loss(reduction="mean")

    def forward(self, x):
        return self.net(x)

    def _step(self, images):
        output = self.net(images)
        output = output.permute(0, 2, 3, 1)  # Height Width Channels

        # calc l1 losses to shifted pixels

        # shift array by 1
        vertical_shifted = output[:, :-1]
        # trim last element to match shifted size
        vertical_trimmed = output[:, 1:]

        # shift array by 1
        horizontal_shifted = output[:, :, :-1]
        # trim last element to match shifted size
        horizontal_trimmed = output[:, :, 1:]

        lhpy = self.connectivity_loss_fn(
            input=vertical_trimmed, target=vertical_shifted
        )
        lhpz = self.connectivity_loss_fn(
            input=horizontal_trimmed, target=horizontal_shifted
        )

        # get "cluster" assignments
        target = torch.argmax(output, 3)

        # loss
        loss_similarity = self.similarity_loss_fn(
            output.reshape(-1, self.n_channels), target.reshape(-1)
        )
        loss_val = (
            self.similarity_weight * loss_similarity
            + self.connectivity_weight * (lhpy + lhpz)
        )

        # calc n labels for each item in batch
        n_labels = [len(torch.unique(target[i])) for i in range(target.shape[0])]
        avg_n_labels = torch.mean(torch.Tensor(n_labels))

        return loss_val, avg_n_labels

    def training_step(self, batch, batch_nb):
        images = batch
        loss_val, avg_n_labels = self._step(images=images)

        self.log(
            "train_loss",
            loss_val,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "avg_n_labels",
            avg_n_labels,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss_val

    @staticmethod
    def _iou_score(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
        pred_classes = torch.unique(pred)
        true_classes = torch.unique(target)
        len(pred_classes), len(true_classes)

        all_ious = torch.zeros((len(pred_classes), len(true_classes)))
        for i, pred_class in enumerate(pred_classes):
            a = (pred == pred_class).int()
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

        return score_a, score_b

    # TODO or test?
    def validation_step(self, batch, batch_idx):
        ims, targets = batch
        targets = targets.cpu()

        out = self.net(ims).detach().cpu()
        preds = out.argmax(dim=1)

        scores = torch.zeros(len(ims), 2)
        for i, (pred, target) in enumerate(zip(preds, targets)):
            scores[i] = torch.FloatTensor(self._iou_score(pred=pred, target=target))

        return scores

    def validation_epoch_end(self, outputs):
        ious = torch.stack(outputs).squeeze(1).mean(dim=0)
        iou_best_truth = ious[0]
        iou_best_pred = ious[1]
        iou_avg = ious.mean()

        self.log(
            "val_iou_best_truth",
            iou_best_truth,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val_iou_best_pred",
            iou_best_pred,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.log(
            "val_iou_avg",
            iou_avg,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        results = {
            "val_iou_best_truth": iou_best_truth,
            "val_iou_best_pred": iou_best_pred,
            "val_iou_avg": iou_avg
        }
        return results

    def configure_optimizers(self):
        return torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        similarity_weight: weight for similarity loss component (default 1.0)
           connectivity_weight: weight for connectivity loss component (defafult 1.0)
           lr: learning (default 0.11)
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--similarity_weight",
            type=float,
            default=1.0,
            help="weight for similarity loss component",
        )
        parser.add_argument(
            "--connectivity_weight",
            type=float,
            default=1.0,
            help="weight for connectivity loss component",
        )
        parser.add_argument("--lr", type=float, default=0.1, help="SGD: learning rate")

        return parser
