import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import nltk
import json
from itertools import groupby


class CharadesInstance:
    def __init__(self, start_frame, end_frame, num_frames, fps):
        self.start_frame = start_frame
        self.end_frame = min(int(end_frame), int(num_frames))
        self.fps = fps

    @property
    def start_time(self):
        return self.start_frame / self.fps

    @property
    def end_time(self):
        return self.end_frame / self.fps


class CharadesVideoRecord:
    def __init__(self, props_data, query_tokens, gt_start_frame, gt_end_frame, fps, gt_start_time,
                 gt_end_time, duration):
        self.props_data = props_data
        self.query_tokens = query_tokens
        self.gt_start_frame = gt_start_frame
        self.gt_end_frame = gt_end_frame
        self.fps = fps
        self.duration = duration
        self.gt_start_time = gt_start_time
        self.gt_end_time = min(gt_end_time, duration)

        self._get_props()

    def _get_props(self):
        props_list = self.props_data[2:]
        num_frames = int(self.props_data[1])
        props_list = [(float(x.strip().split()[0]), float(x.strip().split()[1])) for x in props_list]
        self.proposals = [CharadesInstance(x[0], x[1], num_frames, self.fps) for x in props_list]

    @property
    def id(self):
        return self.props_data[0]

    @property
    def query_length(self):
        return len(self.query_tokens)

    @property
    def num_frames(self):
        return int(self.props_data[1])


class CharadesSTA(Dataset):

    def __init__(self, dataset_configs, split='train', transform=None, ):
        dataset_configs = vars(dataset_configs)
        self.lang_data = list(open(f"./data/dataset/Charades/Charades_sta_{split}.txt", 'r'))
        self.fps_info = json.load(open('./data/dataset/Charades/Charades_fps_dict.json', 'r'))
        self.duration_info = json.load(open('./data/dataset/Charades/Charades_duration.json', 'r'))
        self.word2id = json.load(open('./data/dataset/Charades/Charades_word2id.json', 'r'))
        self.ft_root = dataset_configs[dataset_configs['feature_type']]['feature_root']
        self.ft_window_size = dataset_configs[dataset_configs['feature_type']]['ft_window_size']
        self.ft_overlap = dataset_configs[dataset_configs['feature_type']]['ft_overlap']
        self._load_props_data(dataset_configs["props_file_path"])
        self._preprocess_video_lang_data()

    def _load_props_data(self, props_file_path):
        lines = list(open(props_file_path, 'r'))
        groups = groupby(lines, lambda x: x.startswith('#'))
        info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]
        # props_dict: {vid_name: props_info}
        # props_info:
        # num_frames
        # p1_start_frame, p1_end_frame
        # p2_start_frame, p2_end_frame
        props_dict = {x[0].strip().split()[-1]: x for x in info_list}
        self.props_dit = props_dict

    def _preprocess_video_lang_data(self):
        self.video_list = []
        for item in self.lang_data:
            first_part, query_sentence = item.strip().split('##')
            query_sentence = query_sentence.replace('.', '')
            vid_name, start_time, end_time = first_part.split()
            query_words = nltk.word_tokenize(query_sentence)
            query_tokens = [self.word2id[word] for word in query_words]
            fps = float(self.fps_info[vid_name])
            gt_start_time = float(start_time)
            gt_end_time = float(end_time)
            gt_start_frame = float(start_time) * fps
            gt_end_frame = float(end_time) * fps
            props_data = self.props_dit[vid_name]
            duration = float(self.duration_info[vid_name])
            self.video_list.append(CharadesVideoRecord(props_data, query_tokens, gt_start_frame, gt_end_frame, fps,
                                                       gt_start_time, gt_end_time, duration))

    def get_data(self, video:CharadesVideoRecord):
        '''
        :param video:
        :return: vid_name,
        (all_props_start_frame, all_props_end_frame) : [N, 2], all_props_feature: [N, ft_dim],
        (gt_start_frame, gt_end_frame): [1, 2], query_tokens: [N2, ], query_length: [1,], props_num: [1,]
        '''
        all_proposals = video.proposals
        num_frames = video.num_frames
        vid_name = video.id
        vid_duration = float(self.duration_info[vid_name])
        props_list = []
        props_fts = []
        vid_feature = torch.load(os.path.join(self.ft_root, f"{vid_name}.pt"))

        for i, proposal in enumerate(all_proposals):

            p_start_frame = proposal.start_frame
            p_end_frame = proposal.end_frame
            prop_duration = p_end_frame - p_start_frame
            prop = p_start_frame / num_frames, p_end_frame / num_frames
            props_list.append(prop)
            # ft_start_index = list(range(1, int(p_start_frame)+1, 8))[-1]
            ft_interval = int(self.ft_window_size * (1 - self.ft_overlap))
            ft_start_index = (int(p_start_frame) // ft_interval) * ft_interval

            if prop_duration <= self.ft_window_size:
                # proposal only contains one segment feature
                ft_s_idx = ft_start_index // ft_interval
                ft_indices = [ft_s_idx]

            else:
                # Note: C3D features have dim of (num_frames - window_size) / interval: window_size=16, interval=8
                # indices = range(ft_start_index, p_end_frame if (p_end_frame + self.ft_window_size) <= num_frames
                # else (p_end_frame - self.ft_window_size), ft_interval)
                indices = range(ft_start_index, p_end_frame, ft_interval)
                ft_s_idx = list(map(lambda x: x // ft_interval, indices))
                ft_indices = ft_s_idx
            # if os.path.exists(os.path.join(self.ft_root, f"{vid_name}.pt")):
            #     vid_feature = torch.load(os.path.join(self.ft_root, f"{vid_name}.pt"))
            # else:
            #     props_fts = [torch.zeros([4096]) for _ in range(len(all_proposals))]
            #     break
            ft_indices = sorted(list(map(lambda x: min(len(vid_feature) - 1, x), ft_indices)))
            # if ft_indices[-1] >= len(vid_feature):
            #     ft_indices = sorted(list(map(lambda x: min(len(vid_feature) - 1, x), ft_indices)))
                # ft_indices = sorted(list(filter(lambda x: x < len(vid_feature), ft_indices)))
                # print(f"{vid_name} has less feature:  {len(vid_feature)} / {(num_frames -self.ft_window_size) // ft_interval} ")
            # prop_feature = vid_feature[ft_indices, :].max(dim=0)
            prop_feature = torch.max(vid_feature[ft_indices, :], dim=0)[0]
            props_fts.append(prop_feature)

        props_fts = torch.stack(props_fts)
        props_s_e_list = torch.from_numpy(np.array(props_list))
        # gt_start_frame_normal = video.gt_start_frame / num_frames
        # gt_end_frame_normal = video.gt_end_frame / num_frames
        # time instead of frame
        gt_start_frame_normal = video.gt_start_time / vid_duration
        gt_end_frame_normal  = video.gt_end_time / vid_duration
        gt_s_e = (gt_start_frame_normal, gt_end_frame_normal)
        query_tokens = video.query_tokens
        query_length = len(query_tokens)
        query_tokens = torch.from_numpy(np.array(query_tokens))
        num_props = len(all_proposals)

        return vid_name, props_s_e_list, props_fts, gt_s_e, query_tokens, query_length, num_props, num_frames

    def __getitem__(self, index):
        return self.get_data(self.video_list[index])

    def __len__(self):
        return len(self.video_list)



def collate_data(batch):
    vid_name_list, gt_s_e_list,  q_len_list, props_num_list, num_frames_list = \
    [[] for _ in range(5)]
    data_sorted_by_q_len = sorted(batch, key=lambda x: x[5], reverse=True)
    batch_size = len(batch)
    ft_dim = batch[0][2].size(-1)
    max_props_num = max(map(lambda x: x[6], batch))
    max_query_len = max(map(lambda x: x[5], batch))
    props_features = torch.zeros(batch_size, max_props_num, ft_dim)
    props_s_e = torch.zeros(batch_size, max_props_num, 2, dtype=torch.double)
    query_tokens = torch.zeros(batch_size, max_query_len)

    for i, sample in enumerate(data_sorted_by_q_len):
        vid_name_list.append(sample[0])
        # props_s_e_lists.append(sample[1])
        gt_s_e_list.append(sample[3])
        q_len_list.append(sample[5])
        props_num_list.append(sample[6])
        num_frames_list.append(sample[7])
        # pad query
        query_len = sample[5]
        query_tokens[i, :query_len] = sample[4]
        # pad feature
        props_num = sample[6]
        props_features[i, :props_num, :] = sample[2]
        # pad props_start_end
        props_s_e[i, :props_num, :] = sample[1]

    query_length = torch.LongTensor(np.array(q_len_list))
    query_tokens = query_tokens.long()
    gt_start_end = torch.from_numpy(np.array(gt_s_e_list)).double()
    props_num_list = torch.from_numpy(np.array(props_num_list))
    num_frames_list = torch.from_numpy(np.array(num_frames_list))
    '''
    vid_name_list:  a list with length of batch size
    props_s_e:  a Tensor with size of [batch, max_props_num, 2]
    props_features:  a tensor with size of [batch, max_props_num, ft_dim]
    gt_s_e_list:  a tensor with length of batch size, and each item is a tuple (start_frame/num_frames, end_frame/num_frames)
    query_tokens:  a tensor with size of [batch, max_query_length]
    query_length:  a LongTensor filled with the lengths of each query
    props_num_list:  a tensor filled with the number of proposal in each video
    num_frames_list: a tensor containing the number of frames in each video
    '''
    return vid_name_list, props_s_e, props_features, gt_start_end, query_tokens, query_length, \
           props_num_list, num_frames_list




if __name__ == '__main__':
    root = "/home/datasets/Charades/Charades_MFnet_32unit_stride8_merged/"
    props_file = './utils/Charades_MAN_props.txt'
    dataset = CharadesSTA(root, props_file)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_data, num_workers=1)
    from tqdm import tqdm
    for i, batch in enumerate(tqdm(dataloader)):
        data = batch
        a = 0
