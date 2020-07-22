from tqdm import tqdm
import os
from glob import glob
import argparse
from multiprocessing import Pool




def run(input_video_path):
    video_list = glob(os.path.join(input_video_path, '*.mp4'))
    with Pool(16) as pool:
        pool.map(extract_frames, video_list)



def extract_frames(input_info):
    
    # frames output path
    video_name = os.path.basename(input_info).split('.')[0]
    frames_save_path = os.path.join(output_path, video_name)
    os.makedirs(frames_save_path, exist_ok=True)
    video_path = os.path.join(input_path, f"{video_name}.mp4")
    
    cmd = f'ffmpeg -i {video_path} {frames_save_path}/%05d.jpg  -loglevel 24'
    os.system(cmd)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--split', choices=['train', 'val', 'test'], required=True, help='The name of split')
    parser.add_argument('--input_path', type=str, required=True, help='The input path of videos')
    parser.add_argument('--output_path', type=str, required=True, help='The output path of extracted frames')
    args = parser.parse_args()
    # split_name = args.split
    input_path = args.input_path
    output_path = args.output_path
#     split_path = os.path.join("/mnt/dataset/alvin/data_splits/", f"{split_name}.list")
    
    run(input_path)
    
    
    
