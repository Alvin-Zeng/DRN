import torch

def segment_tiou(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    # calculate interaction
    inter_max_xy = torch.min(box_a[:, -1].unsqueeze(-1).expand(A, B),
                             box_b[:, -1].unsqueeze(0).expand(A, B))
    inter_min_xy = torch.max(box_a[:, 0].unsqueeze(-1).expand(A, B),
                             box_b[:, 0].unsqueeze(0).expand(A, B))
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)

    # calculate union
    union_max_xy = torch.max(box_a[:, -1].unsqueeze(-1).expand(A, B),
                             box_b[:, -1].unsqueeze(0).expand(A, B))
    union_min_xy = torch.min(box_a[:, 0].unsqueeze(-1).expand(A, B),
                             box_b[:, 0].unsqueeze(0).expand(A, B))
    union = torch.clamp((union_max_xy - union_min_xy), min=0)

    iou = inter / (union+1e-6)

    return iou


def merge_segment(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)

    # calculate union
    union_max_xy = torch.max(box_a[:, -1].unsqueeze(-1).expand(A, B),
                             box_b[:, -1].unsqueeze(0).expand(A, B))
    union_min_xy = torch.min(box_a[:, 0].unsqueeze(-1).expand(A, B),
                             box_b[:, 0].unsqueeze(0).expand(A, B))
    union_xy = torch.cat([union_min_xy.unsqueeze(-1), union_max_xy.unsqueeze(-1)], -1)

    return union_xy