import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.language_module import QueryEncoder
from model.graph_module import LanguageGuidedGraphNetwork
from model.backbone import Backbone
from model.basic_blocks import conv_with_kaiming_uniform
from model.fcos import build_fcos
from model.FPN import FPN


class mainModel(nn.Module):
    def __init__(self, vocab_size, dataset_configs, hidden_dim=512, embed_dim=300, bidirection=True,
                 graph_node_features=1024):
        super(mainModel, self).__init__()
        dataset_configs = vars(dataset_configs)
        self.first_output_dim = dataset_configs["first_output_dim"]
        self.fpn_feature_dim = dataset_configs["fpn_feature_dim"]
        self.feature_dim = dataset_configs[dataset_configs['feature_type']]['feature_dim']
        self.query_encoder = QueryEncoder(vocab_size, hidden_dim, embed_dim, dataset_configs["lstm_layers"], bidirection)

        channels_list = [
            (self.feature_dim+256, self.first_output_dim, 3, 1),
            (self.first_output_dim, self.first_output_dim * 2, 3, 2),
            ((self.first_output_dim * 2), self.first_output_dim * 4, 3, 2),
        ]
        conv_func = conv_with_kaiming_uniform(use_bn=True, use_relu=True)
        self.backbone_net = Backbone(channels_list, conv_func)
        self.fpn = FPN([256, 512, 1024], 512, conv_func)
        self.fcos = build_fcos(dataset_configs, self.fpn_feature_dim)
        # self.query_fc = nn.Linear(1024, self.feature_dim)
        self.prop_fc = nn.Linear(self.feature_dim, self.feature_dim)
        self.position_transform = nn.Linear(3, 256)

        for t in range(len(channels_list)):
            if t > 0:
                setattr(self, "qInput%d" % t, nn.Linear(1024, channels_list[t-1][1]))
            else:
                setattr(self, "qInput%d" % t, nn.Linear(1024, self.feature_dim))

    def forward(self, query_tokens, query_length, props_features,
                props_start_end, gt_start_end, props_num, num_frames):

        position_info = [props_start_end, props_start_end]
        position_feats = []
        query_features = self.query_encoder(query_tokens, query_length)
        for i in range(len(query_features)):
            query_fc = getattr(self, "qInput%d" % i)
            query_features[i] = query_fc(query_features[i])
            if i > 1:
                position_info.append(torch.cat([props_start_end[:, :: 2*(i-1), [0]], props_start_end[:, 1:: 2*(i-1), [1]]], dim=-1))
            props_duration = (position_info[i][:, :, 1] - position_info[i][:, :, 0]).unsqueeze(-1)
            position_feat = torch.cat((position_info[i], props_duration), dim=-1).float()
            position_feats.append(self.position_transform(position_feat).permute(0, 2, 1))

        # query_features = query_features.unsqueeze(1).repeat(1, 16, 1)
        # query_features = self.query_fc(query_features)
        props_features = self.prop_fc(props_features)
        # props_duration = (props_start_end[:, :, 1] - props_start_end[:, :, 0]).unsqueeze(-1)
        # position_feat = torch.cat((props_start_end, props_duration), dim=-1).float()
        # # position_feat = self.relu(self.position_transform(position_feat))
        # position_feat = self.position_transform(position_feat)
        # props_features = torch.cat((props_features, position_feat), dim=-1)

        # inputs = torch.rand((32, 768, 16)).cuda()
        inputs = props_features.permute(0, 2, 1)
        outputs = self.backbone_net(inputs, query_features, position_feats)
        outputs = self.fpn(outputs)

        # outputs = [torch.rand((32, 512, 16)).cuda(), torch.rand((32, 512, 8)).cuda(), torch.rand((32, 512, 4)).cuda()]
        # targets = torch.rand((32, 2)).cuda()

        box_lists, loss_dict = self.fcos(outputs, gt_start_end.float())


        # edge_pred, edge_label, node_pred, node_label, iou_gt = self.lang_guided_graph_network(
        #     props_features, query_features, props_start_end, gt_start_end, props_num)

        # return edge_pred, edge_label, node_pred, node_label, iou_gt
        return box_lists, loss_dict
