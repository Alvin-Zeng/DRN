import torch
import torch.nn as nn

class LGP(nn.Module):
    def __init__(self, input_dim=1024, query_dim=1024, use_bn=True):
        super(LGP, self).__init__()

        in_channels = query_dim
        out_channels = input_dim

        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False if use_bn else True
        )

        nn.init.kaiming_uniform_(conv.weight, a=1)

        module = [conv,]

        module.append(nn.BatchNorm1d(input_dim))

        self.query_fc = nn.Sequential(*module)

    def forward(self, inputs, query):
        """
        Arguments:
            query: (batch_size, ft_dim=768)
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """

        t = inputs.size(-1)
        batch_size = inputs.size(0)

        query = self.query_fc(query[:, :, None].repeat(1, 1, t))
        att = inputs * query
        att = att.view(batch_size, -1, int(t / 2), 2)
        att = torch.sum(att, dim=1)
        att = nn.functional.softmax(att, dim=-1)
        att = att.unsqueeze(1)

        outputs = inputs.view(batch_size, -1, int(t/2), 2) * att
        outputs = torch.sum(outputs, dim=-1)

        return outputs

if __name__ == "__main__":

    net = LGP(input_dim=1024, query_dim=1024).cuda()

    inputs = torch.rand(32, 1024, 16).cuda()
    query = torch.rand(32, 1024).cuda()

    output = net(inputs, query)
    print("done")


