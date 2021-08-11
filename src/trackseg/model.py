from argparse import ArgumentParser

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.nn import functional as F


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
        n_clusters: int,
        similarity_weight: float = 1.0,
        connectivity_weight: float = 1.0,
        min_labels: int = 3,
        lr: float = 0.1,
    ):
        """
        Args:
            similarity_weight: weight for similarity loss component (default 1.0)
            connectivity_weight: weight for connectivity loss component (defafult 1.0)
            min_labels: minimum number of labels for segmentation - early exit once reached (default 3)
            lr: learning (default 0.1)
        """
        super().__init__()

        self.save_hyperparameters()

        self.similarity_weight = similarity_weight
        self.connectivity_weight = connectivity_weight
        self.min_labels = min_labels
        self.lr = lr
        self.n_clusters = n_clusters

        self.net = MyNet(input_dim=3, n_channel=n_clusters, n_conv=2)

        self.similarity_loss_fn = nn.CrossEntropyLoss()
        self.connectivity_loss_fn = nn.L1Loss(reduction="mean")

    def forward(self, x):
        return self.net(x)

    def _step(self, batch):
        output = self.net(batch)
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
            output.reshape(-1, self.n_clusters), target.reshape(-1)
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
        loss_val, avg_n_labels = self._step(batch)

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

    def validation_step(self, batch, batch_idx):
        loss_val = self._step(batch)
        return loss_val

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        log_dict = {"val_loss": loss_val}

        # TODO consistency with train step logging
        return {
            "log": log_dict,
            "val_loss": log_dict["val_loss"],
            "progress_bar": log_dict,
        }

    def configure_optimizers(self):
        return torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        similarity_weight: weight for similarity loss component (default 1.0)
           connectivity_weight: weight for connectivity loss component (defafult 1.0)
           min_label: minimum number of labels for segmentation - early exit once reached (default 3)
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
        parser.add_argument(
            "--min_labels",
            type=int,
            default=3,
            help="minimum number of labels for segmentation - early exit once reached",
        )
        parser.add_argument("--lr", type=float, default=0.1, help="SGD: learning rate")

        return parser


def cli_main():
    from track_segmentation.unsupervised_segmentation.datamodule import (
        UnsupervisedSegmentationDataModule,
    )

    seed_everything(1)

    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)
    # model args
    parser = UnsupervisedSemSegment.add_model_specific_args(parser)
    # datamodule args
    parser = UnsupervisedSegmentationDataModule.add_argparse_args(parser)

    args = parser.parse_args()

    # data
    dm = UnsupervisedSegmentationDataModule(args.data_dir).from_argparse_args(args)

    # model
    model = UnsupervisedSemSegment(**args.__dict__)

    # train
    trainer = Trainer().from_argparse_args(args)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()
