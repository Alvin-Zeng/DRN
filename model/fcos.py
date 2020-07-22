import math
import torch
import torch.nn.functional as F
from torch import nn
import pickle
from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg["fcos_num_class"] - 1

        cls_tower = []
        bbox_tower = []
        for i in range(cfg["fcos_conv_layers"]):
            cls_tower.append(nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            cls_tower.append(nn.BatchNorm1d(in_channels))
            cls_tower.append(nn.ReLU())
            # cls_tower.append((nn.Dropout(p=0.5)))
            bbox_tower.append(nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            bbox_tower.append(nn.BatchNorm1d(in_channels))
            bbox_tower.append(nn.ReLU())
            # bbox_tower.append((nn.Dropout(p=0.5)))

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv1d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )

        self.bbox_pred = nn.Conv1d(
            in_channels, 2, kernel_size=3, stride=1,
            padding=1
        )

        self.centerness = nn.Conv1d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        self.mix_fc = nn.Sequential(
            nn.Conv1d(2 * in_channels, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

        self.iou_scores = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels // 2, 1, kernel_size=1, stride=1),
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness, self.iou_scores, self.mix_fc]:
            for l in modules.modules():
                if isinstance(l, nn.Conv1d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg["fcos_prior_prob"]
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(3)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        # innerness = []
        iou_scores = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            # centerness.append(self.centerness(cls_tower))
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(box_tower)
            )))
            mix_feature = self.mix_fc(torch.cat([cls_tower, box_tower], dim=1))
            iou_scores.append(self.iou_scores(mix_feature))
            # iou_scores.append(self.iou_scores(box_tower))
            # innerness.append(self.innerness(box_tower))
        return logits, bbox_reg, centerness, iou_scores


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        head = FCOSHead(cfg, in_channels)
        self.is_first_stage = cfg['is_first_stage']
        box_selector_test = make_fcos_postprocessor(cfg)
        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg["fpn_stride"]

    def forward(self, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness, iou_scores = self.head(features)
        # features = [torch.rand((32, 512, 100, 128)).cuda(), torch.rand((32, 512, 50, 64)).cuda(),
        #             torch.rand((32, 512, 25, 32)).cuda()]
        locations = self.compute_locations(features)

        if self.training:
            return self._forward_train(
                locations, box_cls,
                box_regression,
                targets, iou_scores
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression,
                 targets, iou_scores
            )

    def _forward_train(self, locations, box_cls, box_regression,
                       targets, iou_scores):
        loss_box_cls, loss_box_reg, loss_iou = self.loss_evaluator(
            locations, box_cls, box_regression,  targets, iou_scores, self.is_first_stage
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            'loss_iou': loss_iou,
            # "loss_centerness": loss_centerness,
            # 'loss_innerness': loss_innerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression,
                      targets, iou_scores):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, iou_scores

        )

        loss_box_cls, loss_box_reg, loss_iou = self.loss_evaluator(
            locations, box_cls, box_regression, targets, iou_scores, self.is_first_stage
        )
        pickle.dump(self.loss_evaluator.total_points, open('total_points.pkl', 'wb'))

        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            'loss_iou': loss_iou
            # "loss_centerness": loss_centerness,
            # 'loss_innerness': loss_innerness
        }
        return boxes, losses

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            t = feature.size(-1)
            locations_per_level = self.compute_locations_per_level(
                t, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, t, stride, device):
        shifts_t = torch.arange(
            0, t * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_t = shifts_t.reshape(-1)
        locations = shifts_t + stride / 2
        return locations

    # def compute_locations(self, features):
    #     locations = []
    #     for level, feature in enumerate(features):
    #         h, w = feature.size()[-2:]
    #         locations_per_level = self.compute_locations_per_level(
    #             h, w, self.fpn_strides[level],
    #             feature.device
    #         )
    #         locations.append(locations_per_level)
    #     return locations
    #
    # def compute_locations_per_level(self, h, w, stride, device):
    #     shifts_x = torch.arange(
    #         0, w * stride, step=stride,
    #         dtype=torch.float32, device=device
    #     )
    #     shifts_y = torch.arange(
    #         0, h * stride, step=stride,
    #         dtype=torch.float32, device=device
    #     )
    #     shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    #     shift_x = shift_x.reshape(-1)
    #     shift_y = shift_y.reshape(-1)
    #     locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    #     return locations


def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)

if __name__ == "__main__":


    def compute_locations_per_level(h, w, stride):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    locations = compute_locations_per_level(10, 10, 4)