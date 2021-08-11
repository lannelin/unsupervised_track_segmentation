import torch


def binary_iou(a: torch.Tensor, b: torch.Tensor):
    if a.shape != b.shape:
        raise ValueError("tensors a and b must be the same shape")

    x = a + b
    iou = (x == 2).sum() / (x > 0).sum()

    return iou
