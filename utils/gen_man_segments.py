import os
from tqdm import tqdm
from glob import glob
import json
import numpy as np

SCALES = [1, 2, 4, 8, 16]


def gen_MAN_segments_props(video_path:str):
    '''
    index start from 1; e.g.[1, 65]
    '''
    frames_dict = json.load(open('../data/Charades_frames_info.json', 'r'))
    vid_list = glob(os.path.join(video_path, "*"))
    with open('Charades_MAN_props.txt', 'w') as f:
        num = 0
        for vid_path in tqdm(vid_list):
            vid_name = os.path.basename(vid_path).split('.')[0]
            num_frames = frames_dict[vid_name]
            f.write(f"#\n{vid_name}\n")
            f.write(f"{num_frames}\n")
            for scale in SCALES:
                props_list = []
                indices = np.linspace(1, num_frames, num=scale, endpoint=False, dtype=int).tolist()

                for i, idx in enumerate(indices):
                    if i != len(indices) - 1:
                        item = f"{idx} {indices[i+1]}\n"
                    else:
                        item = f"{idx} {num_frames}\n"

                    # if ((idx // 8) * 8) > list(range(1, num_frames-15, 8))[-1]:
                    #     print(f"idx {((idx // 8) * 8)} / ft: {(((num_frames - 15) // 8) * 8)}")
                    #     num += 1
                    #     continue

                    props_list.append(item)

                f.writelines(props_list)
        print(num)
if __name__ == '__main__':
    video_path = "/home/datasets/Charades/Charades_RGB_Extracted/"
    gen_MAN_segments_props(video_path)