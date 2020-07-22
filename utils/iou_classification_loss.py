from torch import nn
import torch

class Iou_Class_Loss(nn.Module):
    def __init__(self):
        super(Iou_Class_Loss, self).__init__()

    def forward(self, prediction, target):

        total_loss =  target * torch.log(prediction) + (1-target) * torch.log(1-prediction)

        loss = -1 * torch.mean(total_loss)

        return loss

if __name__ == "__main__":

    pred = torch.rand(5, 5, 5, 2)

    target = torch.rand(5, 5, 5, 1)
    target1 = torch.cat([target, 1-target], -1)

    pred2 = pred.view(-1, 1)

    target2 = target.view(-1, 1)

    # criterion = Iou_Class_Loss()

    criterion2 = nn.BCELoss()

    loss = criterion2(pred, target)
    loss1 = criterion2(pred, target1)

    loss2 = criterion2(pred2, target)
    loss21 = criterion2(pred2, target)

    print(loss, loss1)