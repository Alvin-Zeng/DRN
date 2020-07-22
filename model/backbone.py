import torch
import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self, channels_list, conv_block):
        super(Backbone, self).__init__()

        self.num_layers = len(channels_list)
        self.blocks = []
        for idx, channels_config in enumerate(channels_list):
            block = "forward_conv{}".format(idx)
            block_module = conv_block(channels_config[0], channels_config[1],
                                      kernel_size=channels_config[2], stride=channels_config[3])
            self.add_module(block, block_module)
            self.blocks.append(block)

    def forward(self, x, query_fts, position_fts):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        results = []

        for idx in range(self.num_layers):
            query_ft = query_fts[idx].unsqueeze(1).permute(0, 2, 1)
            position_ft = position_fts[idx]
            x = query_ft * x
            if idx == 0:
                x = torch.cat([x, position_ft], dim=1)
            x = self._modules['forward_conv{}'.format(idx)](x)
            results.append(x)

        return tuple(results)