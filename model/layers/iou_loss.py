import torch
from torch import nn


class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_right = pred[:, 1]

        target_left = target[:, 0]
        target_right = target[:, 1]

        intersect = torch.min(pred_right, target_right) + torch.min(pred_left, target_left)
        target_area = target_left + target_right
        pred_area = pred_left + pred_right
        union = target_area + pred_area - intersect

        losses = -torch.log((intersect + 1e-8) / (union + 1e-8))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()
