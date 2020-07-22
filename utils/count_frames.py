from tqdm import tqdm
from glob import glob
import os
import sys
import json
from itertools import groupby
from functools import reduce
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from concurrent.futures import  ProcessPoolExecutor, as_completed

def count_frames(video_root:str):
    video_list = glob(os.path.join(video_root, '*.mp4'))
    for vid in tqdm(video_list):
        cmd = f"ffprobe -v error \
        -select_streams v:0      \
        -show_entries            \
        stream=nb_frames         \
        -of default=nokey=1:noprint_wrappers=1  \
        {vid} 2>&1 | tee -a "
        print(vid)
        sys.stdout.flush()
        os.system(cmd)


def convert_info2dict(info_file:str):
    info_data = list(open(info_file))
    info_dict = {}
    for i, line in enumerate(tqdm(info_data)):
        if not (i%2):
            vid_name = os.path.basename(line.strip()).split('.')[0]
            num_frames = int(info_data[i+1].strip())
            info_dict[vid_name] = num_frames
    
    json.dump(info_dict, open("Charades_frames_info.json", 'w'), indent=4)


def count_proposal(props_data_file):
    props_data = list(open(props_data_file))
    groups = groupby(props_data, lambda x: x.startswith('#'))
    info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]
    num_dict = {}
    for info in tqdm(info_list):
        props = info[2:]
        if len(props) == 0:
            a = 0
        vid_name = info[0]
        if vid_name not in num_dict:
            num_dict[vid_name] = len(props)

    props_num = list(num_dict.values())
    props_num.sort()
    max_num = props_num[-1]
    min_num = props_num[0]
    mean_num = reduce(lambda x, y: x+y, props_num)/len(props_num)
    print(f"max {max_num}; min {min_num}; mean: {mean_num}")


def count_positive_negtive_sample_ratio(props_file_path:str, gt_train_file_path:str, gt_test_file_path:str,
                                        fps_dict_path:str):
    fps_dict = json.load(open(fps_dict_path, 'r'))
    gt_train_data = list(open(gt_train_file_path, 'r'))
    gt_test_data = list(open(gt_test_file_path, 'r'))

    lines = list(open(props_file_path, 'r'))
    groups = groupby(lines, lambda x: x.startswith('#'))
    info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]
    props_dict = {x[0].strip().split()[-1]: x for x in info_list}

    def temporal_iou(span_A, span_B):
        """
        Calculates the intersection over union of two temporal "bounding boxes"

        span_A: (start, end)
        span_B: (start, end)
        """
        union = min(span_A[0], span_B[0]), max(span_A[1], span_B[1])
        inter = max(span_A[0], span_B[0]), min(span_A[1], span_B[1])

        if inter[0] >= inter[1]:
            return 0
        else:
            return float(inter[1] - inter[0]) / float(union[1] - union[0])

    def count_pos_neg_ratio(gt_data:list, split_name='Train'):

        train_positive_num = 0
        train_negtive_num = 0
        total_props = 0

        for line in tqdm(gt_data):
            vid_name = line.strip().split()[0]
            fps = float(fps_dict[vid_name])
            gt_start, gt_end = line.strip().split('##')[0].split()[-2:]
            gt_span = (float(gt_start) * fps, float(gt_end) * fps)
            props_info = props_dict[vid_name][2:]
            total_props += len(props_info)
            props_span_list = list(map(lambda x:(float(x.strip().split()[0]), float(x.strip().split()[1])), props_info))
            for prop_span in props_span_list:
                if temporal_iou(prop_span, gt_span) > 0.5:
                    train_positive_num += 1
                else:
                    train_negtive_num += 1

        print(f"{split_name}: \n positive:{train_positive_num / len(gt_data)} \n negative； {train_negtive_num / len(gt_data)} \n ratio: "
              f"{train_positive_num/ train_negtive_num}")

    count_pos_neg_ratio(gt_train_data, 'Train')
    count_pos_neg_ratio(gt_test_data, 'test')
    count_pos_neg_ratio(gt_train_data+gt_test_data, 'Total')

def get_video_list(video_root:str):
    video_list = glob(os.path.join(video_root, '*.mp4'))
    # video_list = list(map(lambda x:os.path.basename(x.strip()), video_list))
    return video_list

def  count_duration(vid_path):
    duration_dict = dict()
    clip = VideoFileClip(vid_path)
    duration = clip.duration
    vid_name = os.path.basename(vid_path).split('.')[0]
    duration_dict[vid_name] = duration
    with open(f'./temp/{vid_name}.json', 'w') as f:
        json.dump(duration_dict, f)
    print(f"{vid_name} counted done!")


def merge_json_file(file_root:str):
    '''
    I have not learned how to write a singe file in a multiprocessing way.
    So, this function serves to merging separated files.
    '''
    total_dict = {}
    file_list = glob(os.path.join(file_root, '*.json'))
    for file_path in tqdm(file_list):
        curr_dict = json.load(open(file_path, 'r'))
        total_dict.update(curr_dict)
        # cmd = f'rm {curr_dict}'
        # os.system(cmd)

    json.dump(total_dict, open('./Charades_duration.json', 'w'), indent=4)


def count_rest_video(file_root:str, video_root:str):
    already_list = glob(os.path.join(file_root, '*.json'))
    already_list = [os.path.basename(name).split('.')[0] for name in already_list]
    vid_path_list = glob(os.path.join(video_root, '*.mp4'))
    vid_name_list = [os.path.basename(vid_path).split('.')[0] for vid_path in vid_path_list]
    rest_list = [vid_name for vid_name in vid_name_list if vid_name not in already_list ]
    rest_vid_path = [os.path.join(video_root, f"{vid_name}.mp4") for vid_name in rest_list]
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(count_duration, vid_name) for vid_name in tqdm(rest_vid_path)]
        for future in as_completed(futures):
            if not future.result:
                raise RuntimeError



def count_GT_duration(train_file_path:str):
    from collections import Counter
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.ticker import PercentFormatter
    duration_info = json.load(open('../data/Charades_duration.json', 'r'))
    train_data = list(open(train_file_path, 'r'))
    result = []
    for line in tqdm(train_data):
        first_part, second_part = line.strip().split('##')
        vid_name, gt_start_time, gt_end_time = first_part.strip().split()
        gt_start_time = float(gt_start_time)
        gt_end_time = float(gt_end_time)
        video_length = duration_info[vid_name]
        duration = gt_end_time - gt_start_time
        duration_ratio = (duration / video_length) * 16
        result.append(duration_ratio)
    counter = Counter(result)
    x = list(counter.keys())
    # y = list(counter.values())
    n_bins = len(x)
    fig, ax = plt.subplots(tight_layout=True)
    plt.xlim(0, 16)
    ax.hist(result, bins=n_bins // 10)
    plt.show()



if __name__ == "__main__":
    # count_frames("/home/datasets/Charades/Charades_v1_480/")
    # convert_info2dict('./Charades_frames_info.txt')
    # count_proposal('../data/Charades_sw_props.txt')
    # count_positive_negtive_sample_ratio('../data/Charades_sw_props.txt', '../data/charades_sta_train.txt',
    #                                     '../data/charades_sta_test.txt', '../data/Charades_fps_dict.json')
    # video_root = "/home/datasets/Charades/Charades_v1_480/"
    # file_root = "/home/xuhaoming/Projects/cvpr_baseline/One-stage-moment-retrieval/utils/temp/"
    # count_rest_video(file_root, video_root)
    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     futures = [executor.submit(count_duration, vid_name) for vid_name in tqdm(get_video_list(video_root))]
    #     for future in as_completed(futures):
    #         if not future.result:
    #             raise RuntimeError
    # count_duration(get_video_list(video_root)[0])
    # merge_json_file(file_root)
    count_GT_duration("/home/xuhaoming/Projects/cvpr_baseline/One-stage-moment-retrieval/data/charades_sta_train.txt")

        