import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from itertools import groupby


def merge_ft(input_path, output_path):
    ft_raw = sorted(os.listdir(input_path))
    ft_group = groupby(ft_raw, lambda s: s.split('_')[0])
    ft_list = [[x for x in list(vid_ft)] for vid_name, vid_ft in ft_group]
    
    for vid_ft in tqdm(ft_list):
        vid_name = vid_ft[0].split('_')[0]
        ft_concat = []
        for ft in sorted(vid_ft, key=lambda s: float(s.split('_')[1])):
            ft_concat.append(np.load(os.path.join(input_path, ft)))
        ft_concat = torch.from_numpy(np.array(ft_concat))
        torch.save(ft_concat, os.path.join(output_path, vid_name + '.pt'))

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    merge_ft(input_path, output_path)
