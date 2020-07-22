import csv
import os
import json
from tqdm import tqdm
reader = csv.reader(open('./Charades_fps_size_info.csv', 'r'))
output_dict = {}
for row in tqdm(reader):
    vid_name = row[1].split('.')[0]
    output_dict[vid_name] = row[3].strip().split()[0]
json.dump(output_dict, open('Charades_fps_dict.json', 'w'), indent=4)

