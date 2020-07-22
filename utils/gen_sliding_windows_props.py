import pickle
import os
from tqdm import tqdm
from glob import glob
import json

SCALES = [64, 128, 256, 512]
def gen_sw_props(video_path:str):
    '''
    index start from 1; e.g.[1, 65]
    '''
    frames_dict = json.load(open('data/Charades_frames_info.json', 'r'))
    vid_list = glob(os.path.join(video_path, "*"))
    with open('Charades_sw_props.txt', 'w') as f:
        for vid_path in tqdm(vid_list):
            vid_name = os.path.basename(vid_path).split('.')[0]
            num_frames = frames_dict[vid_name]
            f.write(f"#\n{vid_name}\n")
            f.write(f"{num_frames}\n")
            for scale in SCALES:
                interval = int(scale * 0.2)
                start_list = list(range(1, num_frames, interval))
                props_list = list(map(lambda x: f"{x} {x+scale}\n", start_list))
                # filter out those that are out of total frames
                props_list = list(filter(lambda x:int(x.strip().split()[-1]) <= num_frames, props_list))
                f.writelines(props_list)








if __name__  ==  '__main__':
    gen_sw_props("/home/datasets/Charades/Charades_v1_480/")


