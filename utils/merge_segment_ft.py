from glob import glob
from tqdm import tqdm
import os
import json
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


def merge_segment_ft(vid_name):
    segment_fts_list = glob(os.path.join(feature_root, vid_name, "*"))
    ft_num_dict[vid_name] = len(segment_fts_list)
    ft_list = []
    segment_fts_list = sorted(segment_fts_list, key=lambda x: float(os.path.basename(x).split('_')[1]))
    # import ipdb;ipdb.set_trace()
    for i, segment_ft in enumerate(tqdm(segment_fts_list)):
        segment_ft_name = os.path.basename(segment_ft)
        prefix = float(os.path.basename(segment_fts_list[i-1]).split('_')[1]) if i > 0 else 1.0
        now_idx = float(segment_ft_name.split('_')[1])
        next_idx = float(os.path.basename(segment_fts_list[i+1]).split('_')[1]) if i != (len(segment_fts_list)-1) else \
        float(os.path.basename(segment_fts_list[i]).split('_')[1])
        assert prefix <= now_idx <= next_idx
        ft = torch.load(segment_ft)
        ft_list.append(ft)
    merged_feature = torch.stack(ft_list)
    torch.save(merged_feature, f"/home/datasets/Charades/Charades_MFnet_32unit_stride8_merged/{vid_name}.pt")


def gen_vid_name_list(vid_path):
    vid_name_list = glob(os.path.join(vid_path, '*mp4'))
    vid_name_list = list(map(lambda x: os.path.basename(x).split('.')[0], vid_name_list))
    return vid_name_list


if __name__ == '__main__':
    ft_num_dict = {}
    feature_root = "/home/datasets/Charades/Charades_MFnet_32unit_stride8/"
    vid_path = "/home/datasets/Charades/Charades_v1_480/"
    output_path = "/home/datasets/Charades/Charades_MFnet_32unit_stride8_merged/"
    os.makedirs(output_path, exist_ok=True)
    # merge_segment_ft(feature_root, vid_path)
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(merge_segment_ft, vid_name) for vid_name in gen_vid_name_list(vid_path)]
        for future in as_completed(futures):
            if not future.result:
                raise RuntimeError
    # merge_segment_ft(gen_vid_name_list(vid_path)[0])
    json.dump(ft_num_dict, open('mfnet_num_seg_info.json', 'w'), indent=4)

