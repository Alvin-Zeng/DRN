# from torchtools import *
from collections import OrderedDict
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.detection_metrics import segment_tiou, merge_segment
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class InteractionNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features=1):
        super(InteractionNetwork, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.concate_net = nn.Sequential(nn.Linear(in_features=self.in_features * 2,
                                                   out_features=self.in_features,
                                                   bias=True),
                                         nn.ReLU())
        # self.interaction_net = nn.Sequential(nn.Linear(in_features=self.in_features * 3,
        #                                                out_features=self.hidden_features,
        #                                                bias=True),
        #                                      nn.ReLU(),
        #                                      nn.Linear(in_features=self.hidden_features,
        #                                                out_features=self.out_features,
        #                                                bias=True),
        #                                      nn.Sigmoid())
        self.interaction_net = nn.Sequential(nn.Linear(in_features=self.in_features * 3,
                                                       out_features=self.hidden_features,
                                                       bias=True),
                                             nn.Tanh(),
                                             nn.Linear(in_features=self.hidden_features,
                                                       out_features=self.out_features,
                                                       bias=True),
                                             )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, vis_feat, text_feat):
        concate_feat = self.concate_net(torch.cat([vis_feat, text_feat], -1))
        interaction_score = self.interaction_net(
            torch.cat([vis_feat * text_feat, vis_feat + text_feat, concate_feat], -1))
        interaction_score = self.softmax(interaction_score)

        return interaction_score


class NodeRelationNetwork(nn.Module):
    def __init__(self,
                 text_features,
                 node_features,
                 hidden_features,
                 out_features=1,
                 position_dim=3,
                 ratio=[1, 1],
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(NodeRelationNetwork, self).__init__()

        # set options
        self.separate_dissimilarity = separate_dissimilarity
        # set size
        self.text_features = text_features
        self.hidden_features = hidden_features
        self.node_features = node_features
        self.out_features = out_features
        self.dropout = dropout
        self.position_dim = position_dim

        # layers
        # self.visual_w = nn.Sequential(nn.Linear(in_features=self.node_features,
        #                                         out_features=self.hidden_features,
        #                                         bias=True),
        #                               nn.ReLU(inplace=True))
        self.visual_w = nn.Sequential(nn.Conv2d(in_channels=self.node_features,
                                                 out_channels=self.hidden_features,
                                                 kernel_size=1,
                                                 bias=False),
                                       nn.BatchNorm2d(num_features=self.hidden_features),
                                       nn.ReLU(inplace=True))
        # self.textual_w = nn.Sequential(nn.Linear(in_features=self.text_features,
        #                                         out_features=self.hidden_features,
        #                                         bias=True),
        #                               nn.ReLU(inplace=True))
        self.textual_w = nn.Sequential(nn.Conv2d(in_channels=self.text_features,
                                                 out_channels=self.hidden_features,
                                                 kernel_size=1,
                                                 bias=False),
                                       nn.BatchNorm2d(num_features=self.hidden_features),
                                       nn.ReLU(inplace=True))

        self.position_transform = nn.Linear(self.position_dim, self.hidden_features)
        self.relu = nn.LeakyReLU(inplace=True)
        self.num_features_list = [self.hidden_features * r for r in ratio]
        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            if l != (len(self.num_features_list) -1) :
            # set layer
                layer_list['conv{}'.format(l)] = nn.Conv2d(
                    in_channels=self.num_features_list[l-1] if l > 0 else self.node_features + self.hidden_features,
                    out_channels=self.num_features_list[l],
                    kernel_size=1,
                    bias=False)

                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l])

                layer_list['relu{}'.format(l)] = nn.LeakyReLU()

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)
            else:
                # set layer
                layer_list['conv{}'.format(l)] = nn.Conv2d(
                    in_channels=self.num_features_list[l - 1] if l > 0 else self.node_features + self.hidden_features,
                    out_channels=self.num_features_list[l],
                    kernel_size=1,
                    bias=False)

                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l])

                layer_list['relu{}'.format(l)] = nn.Tanh()

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.sim_network = nn.Sequential(layer_list)

        if self.separate_dissimilarity:
            # layers
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                # set layer
                layer_list['conv{}'.format(l)] = nn.Conv2d(
                    in_channels=self.num_features_list[l-1] if l > 0 else self.node_features + self.hidden_features,
                    out_channels=self.num_features_list[l],
                    kernel_size=1,
                    bias=False)
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l])

                layer_list['relu{}'.format(l)] = nn.LeakyReLU()

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=self.dropout)

            layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                               out_channels=1,
                                               kernel_size=1)
            self.dsim_network = nn.Sequential(layer_list)



    def forward(self, node_feat, text_feat, props_s_e):
        # position feature
        props_duration = (props_s_e[:, :, 1] - props_s_e[:, :, 0]).unsqueeze(-1)
        position_feat = torch.cat((props_s_e, props_duration), dim=-1).float()
        position_feat = self.position_transform(position_feat)
        node_feat = torch.cat((node_feat, position_feat), dim=-1)
        # obtain x_i, x_j
        x_i = node_feat
        t_i = text_feat

        x_i = torch.cat([x_i, (self.visual_w(x_i.transpose(1, 2).unsqueeze(-1))
                         * self.textual_w(t_i.transpose(1, 2).unsqueeze(-1))).transpose(1, 2).squeeze(-1).contiguous()], dim=-1)

        x_i = x_i.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)

        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)

        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        # sim_val = torch.sigmoid(self.sim_network(x_ij))
        sim_val = self.softmax(self.sim_network(x_ij))

        if self.separate_dissimilarity:
            dsim_val = torch.sigmoid(self.dsim_network(x_ij))
        else:
            dsim_val = 1.0 - sim_val

        return sim_val, dsim_val



class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 node_features,
                 hidden_features,
                 ratio=[2, 2],
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.node_features = node_features
        self.hidden_features = hidden_features
        self.num_features_list = [self.hidden_features * r for r in ratio]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):

            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.node_features * 3,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # get size
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)

        # get eye matrix (batch_size x 2 x node_size x node_size)
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 2, 1, 1).cuda()

        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)

        # compute attention and aggregate
        # cat([(bs, 1, num, num), (bs, 1, num, num)], 2)
        aggr_feat = torch.bmm(torch.cat(torch.split(edge_feat, 1, 1), 2).squeeze(1), node_feat)
        # aggr_feat: (bs, 2 x node_size, feat_dim)

        # c = aggr_feat.split(num_data, 1)
        # a = torch.cat(c, -1)
        # b = torch.cat([node_feat, a], -1)
        node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1).transpose(1, 2)
        # node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1)
        # node_feat: (bs, node_size, feat_dim * 3)

        # non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        # node_feat: (bs, node_size, feat_dim)

        return node_feat


class EdgeUpdateNetwork(nn.Module):
    def __init__(self,
                 text_features,
                 node_features,
                 hidden_features,
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(EdgeUpdateNetwork, self).__init__()
        # set size
        self.text_features = text_features
        self.hidden_features = hidden_features
        self.node_features = node_features
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout

        self.relation_net = NodeRelationNetwork(node_features=self.node_features + self.hidden_features,
                                                text_features=self.text_features,
                                                hidden_features=self.hidden_features)


    def forward(self, node_feat, edge_feat, text_feat, props_s_e):

        sim_val, dsim_val = self.relation_net(node_feat, text_feat, props_s_e)

        diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 2, 1, 1).cuda()
        edge_feat = edge_feat * diag_mask
        merge_sum = torch.sum(edge_feat, -1, True)
        # set diagonal as zero and normalize
        edge_feat = F.normalize(torch.cat([sim_val, dsim_val], 1) * edge_feat, p=1, dim=-1) #* merge_sum
        force_edge_feat = torch.cat((torch.eye(node_feat.size(1)).unsqueeze(0),
                                     torch.zeros(node_feat.size(1),
                                                 node_feat.size(1)).unsqueeze(0)),
                                    0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).cuda()
        # force edge feat: set the edge value of self-connection to 1
        edge_feat = edge_feat + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1).repeat(1, 2, 1, 1)

        return edge_feat


class GraphNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 text_features,
                 node_features,
                 hidden_features,
                 num_layers,
                 node_dropout=0.0,
                 edge_dropout=0.0):
        super(GraphNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.text_features = text_features
        self.node_features = node_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout

        # self.fc = nn.Sequential(nn.Linear(in_features=self.node_features+self.text_features,
        #                                   out_features=self.hidden_features,
        #                                   bias=True),
        #                         nn.ReLU(),
        #                         nn.Linear(in_features=self.hidden_features,
        #                                   out_features=1,
        #                                   bias=True),
        #                         nn.Sigmoid())
        self.interaction_net = InteractionNetwork(in_features=self.node_features,
                                                  hidden_features=self.hidden_features)

        # for each layer
        for l in range(self.num_layers):
            # set edge to node
            edge2node_net = NodeUpdateNetwork(node_features=self.in_features if l == 0 else self.node_features,
                                              hidden_features=self.hidden_features,
                                              dropout=self.node_dropout if l < self.num_layers-1 else 0.0)

            # set node to edge
            node2edge_net = EdgeUpdateNetwork(text_features=self.text_features,
                                              node_features=self.node_features,
                                              hidden_features=self.hidden_features,
                                              separate_dissimilarity=False,
                                              dropout=self.edge_dropout if l < self.num_layers-1 else 0.0)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)

    # forward
    def forward(self, node_feat, edge_feat, text_feat, props_s_e, num_props):
        # node_feat: (batch_size x num_samples x feat_size)
        # edge_feat: (batch_size x 2 x num_samples x num_samples)
        # text_feat: (batch_size x num_samples x feat_size)

        bs = node_feat.size(0)
        max_prop_num = node_feat.size(1)

        # generate mask for node and edge
        mask_node = torch.arange(max_prop_num, dtype=num_props.dtype,
                                 device=num_props.device).expand(bs, max_prop_num) < num_props.unsqueeze(1)
        mask_edge_temp = mask_node.clone()
        mask_edge_temp2 = mask_node.clone()
        mask_edge = (mask_edge_temp.unsqueeze(1) * mask_edge_temp2.unsqueeze(-1)).unsqueeze(1).repeat(1, 2, 1, 1).float()
        mask_node = mask_node.float().unsqueeze(-1)
        # for each layer
        edge_feat_list = []
        node_feat_list = []
        for l in range(self.num_layers):
            # (1) edge to node
            node_feat = node_feat * mask_node
            node_feat = self._modules['edge2node_net{}'.format(l)](node_feat, edge_feat)

            # (2) node to edge
            edge_feat = edge_feat * mask_edge
            edge_feat = self._modules['node2edge_net{}'.format(l)](node_feat, edge_feat, text_feat, props_s_e)

            # save edge feature
            edge_feat_list.append(edge_feat)
            # save node feature
            node_feat_list.append(self.interaction_net(node_feat, text_feat))
            # node_feat_list.append(self.fc(torch.cat([node_feat, text_feat], -1)))

        return edge_feat_list, node_feat_list



class LanguageGuidedGraphNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 text_features,
                 node_features,
                 hidden_features,
                 num_layers,
                 position_dim=3,
                 pos_thr=0.5,
                 neg_thr=0.1,
                 node_dropout=0.0,
                 edge_dropout=0.0,
                 ):
        super(LanguageGuidedGraphNetwork, self).__init__()

        self.pos_thr = pos_thr
        self.neg_thr = neg_thr

        self.graph_net = GraphNetwork(in_features=in_features+hidden_features,
                                      text_features=text_features,
                                      node_features=node_features,
                                      hidden_features=hidden_features,
                                      num_layers=num_layers,
                                      node_dropout=node_dropout,
                                      edge_dropout=edge_dropout).cuda()

        self.relation_net = NodeRelationNetwork(text_features=text_features,
                                                node_features=in_features + hidden_features,
                                                hidden_features=hidden_features)
        self.position_transform = nn.Linear(position_dim, hidden_features)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, node_feat, text_feat, props_s_e, gt_s_e, num_props):

        bs = props_s_e.size(0)
        max_prop_num = props_s_e.size(1)
        # init with iou mask
        # iou_list = [segment_tiou(props_s_e[n_bs], props_s_e[n_bs]) for n_bs in range(bs)]
        # iou_mask = torch.stack(iou_list).unsqueeze(1).float()

        edge_1, edge_2 = self.relation_net(node_feat, text_feat, props_s_e)
        edge_feat = torch.cat([edge_1, edge_2], 1)
        # edge_feat = torch.cat([edge_1*iou_mask, edge_2*iou_mask], 1)

        # add node feature into graph
        props_duration = (props_s_e[:, :, 1] - props_s_e[:, :, 0]).unsqueeze(-1)
        position_feat = torch.cat((props_s_e, props_duration), dim=-1).float()
        # position_feat = self.relu(self.position_transform(position_feat))
        position_feat = self.position_transform(position_feat)
        node_feat = torch.cat((node_feat, position_feat), dim=-1)

        edge_feat_list, node_feat_list = self.graph_net(node_feat, edge_feat, text_feat, props_s_e, num_props)

        # calculate target
        # calculate iou before merge
        iou_gt_before_list = [segment_tiou(gt_s_e[n_bs].unsqueeze(0), props_s_e[n_bs]) for n_bs in range(bs)]
        iou_gt_before = torch.cat(iou_gt_before_list, 0)

        # calculate new boundary after merge
        merged_props_s_e_list = [merge_segment(props_s_e[n_bs], props_s_e[n_bs]) for n_bs in range(bs)]
        merged_props_s_e = torch.stack(merged_props_s_e_list)

        # calculate iou after merge
        iou_gt_after_list = [segment_tiou(gt_s_e[n_bs].unsqueeze(0),
                                          merged_props_s_e[n_bs].view(-1, 2)).view(max_prop_num, max_prop_num)
                             for n_bs in range(bs)]
        iou_gt_after = torch.stack(iou_gt_after_list)

        # obtain edge label
        iou_gt_before_mat = iou_gt_before.unsqueeze(-1).repeat(1, 1, max_prop_num)
        # iou_gt_before: (bs, max_prop_num, max_prop_num)
        pos_label_ind = iou_gt_after > iou_gt_before_mat
        # neg_label_ind = iou_gt_before <= 0.001
        # keep_label_ind = pos_label_ind * (1-neg_label_ind.unsqueeze(-1).repeat(1,1,max_prop_num))
        # obtain node label
        pos_proposal_ind = iou_gt_before >= self.pos_thr
        # neg_proposal_ind = iou_gt_before < self.pos_thr

        node_label_ind = iou_gt_before > self.pos_thr

        return edge_feat_list, pos_label_ind, node_feat_list, node_label_ind, iou_gt_before * pos_proposal_ind.double()


if __name__ == "__main__":

    model = LanguageGuidedGraphNetwork(in_features=4096,
                                       text_features=1024,
                                       node_features=1024,
                                       hidden_features=512,
                                       num_layers=3).cuda()

    bs = 3
    num_prop = 3
    node_feat = torch.rand(bs, num_prop, 4096).cuda()
    edge_feat = torch.rand(bs, 2, num_prop, num_prop).cuda()
    text_feat = torch.rand(bs, num_prop, 1024).cuda()

    props_s_e = torch.rand(bs, num_prop, 2).cuda()
    gt_s_e = torch.rand(bs, 2).cuda()
    props_num = torch.LongTensor((2, 3, 3)).cuda()

    edge_pred, edge_label, \
    node_pred, node_label, iou_gt = model(node_feat, text_feat, props_s_e, gt_s_e, props_num)

    valid_edge_pred_list = []
    valid_node_pred_list = []
    valid_edge_label_list = []
    valid_node_label_list = []
    for n_bs in range(bs):
        valid_edge_pred_list.append(edge_pred[-1][n_bs, :, :props_num[n_bs], :props_num[n_bs]].permute(1, 2, 0).contiguous().view(-1, 2))
        valid_edge_label_list.append(edge_label[n_bs, :props_num[n_bs], :props_num[n_bs]].contiguous().view(-1))

        valid_node_pred_list.append(node_pred[-1][n_bs, :props_num[n_bs]])
        valid_node_label_list.append(node_label[n_bs, :props_num[n_bs]])

    valid_edge_pred = torch.cat(valid_edge_pred_list, 0)
    valid_edge_label = torch.cat(valid_edge_label_list, 0)

    valid_node_pred = torch.cat(valid_node_pred_list, 0)
    valid_node_pred = torch.cat([valid_node_pred, 1-valid_node_pred], -1)
    valid_node_label = torch.cat(valid_node_label_list, 0)

    print("done")