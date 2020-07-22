import os
from tqdm import tqdm
import json
import torch
import numpy as np
import operator
import pickle
import time
from collections import defaultdict
from copy import deepcopy
import plotly.graph_objects as go

class PostProcessRunner:
    def __init__(self, raw_results):
        self.raw_results = raw_results if isinstance(raw_results, dict) else json.load(open(raw_results, 'r'))
        self.word2id = json.load(open("./data/dataset/Charades/Charades_word2id.json", 'r'))
        # self.word2id = json.load(open("./data/dataset/TACoS/TACoS_word2id_glove_lower.json", 'r'))

    def _postprocess_raw_results(self, update_score=False, score_weight=1):
        process_results_dict = {}
        for vid_name, vid_results in tqdm(self.raw_results.items()):
            for query_result in vid_results:
                query = query_result['query']
                gt = query_result['gt']
                node_predictions = torch.from_numpy(np.array(query_result['node_predictions']))
                edge_predictions = torch.from_numpy(np.array(query_result['edge_predictions']))
                num_props = edge_predictions.size(0)
                update_props_list = []
                for i in range(num_props):
                    # (start, end, match_scores)
                    prop = node_predictions[i]
                    related_props = []
                    # select neighbor proposal(iou > 0)
                    for j in range(num_props):
                        if i != j:
                            neighbor_prop = node_predictions[j]
                            iou = self.temporal_iou(prop, neighbor_prop)
                            if iou > 0:
                                neighbor_prop = neighbor_prop.numpy().tolist()
                                edge_score = edge_predictions[i, j, 0].item()
                                # (start, end, match_score, edge_score)
                                neighbor_prop += [edge_score]
                                related_props.append(neighbor_prop)
                    related_props = sorted(related_props, key=lambda x: x[-1], reverse=True)
                    # merge props
                    most_related_props = related_props[0]
                    updated_prop = self.merge_two_proposals(prop, most_related_props, update_score, score_weight)
                    update_props_list.append(updated_prop)
                    update_props_list = sorted(update_props_list, key=lambda x: x[-1], reverse=True)
                temp_dict = {
                    'query': query,
                    'gt': gt,
                    'node_predictions': update_props_list
                }
                try:
                    process_results_dict[vid_name].append(temp_dict)
                except KeyError:
                    process_results_dict[vid_name] = []
                    process_results_dict[vid_name].append(temp_dict)

        # only save specific 100 items
        visual_data_index = pickle.load(open('/home/xuhaoming/Projects/CVPR2020/utils/visual_data_index.pkl', 'rb'))
        visual_data = {}
        for idx in visual_data_index:
            visual_data[idx] = process_results_dict[idx]

        results_folder = './results/Evaluate/Raw_results'
        os.makedirs(results_folder, exist_ok=True)
        date = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
        json.dump(visual_data, open(os.path.join(results_folder, f'processed_results_{date}.json'), 'w'),
                  indent=4)

        self.processed_results = process_results_dict
        # is_update_score = "no_update_scores" if not self.update_score else f"update_scores_{self.score_weight}"
        # os.makedirs(self.processed_results_output_path, exist_ok=True)
        # save_name = os.path.join(f'processed_{input_file_name}_{is_update_score}.json')
        # json.dump(process_results_dict, open(save_name, 'w'), indent=4)

    def merge_two_proposals(self, prop_a, prop_b, update_score, score_weight):
        start = min(prop_a[0].item(), prop_b[0])
        end = max(prop_a[1].item(), prop_b[1])
        if update_score:
            # update score of prop_a
            match_score = score_weight * prop_a[-1].item() + (1 - score_weight) * prop_b[-1]
        else:
            match_score = prop_a[-1].item()
        merged_props = [start, end, match_score]

        return merged_props

    def _postprocess_raw_results_no_merge(self, save_processed_res):
        process_results_dict = {}
        for vid_name, vid_results in tqdm(self.raw_results.items()):
            for query_result in vid_results:
                query = query_result['query']
                gt = query_result['gt']
                node_predictions = list(sorted(query_result['node_predictions'], key=lambda x: x[-1], reverse=True))
                temp_dict = {
                    'query': query,
                    'gt': gt,
                    'node_predictions': node_predictions,
                    'level': query_result['level']
                }
                try:
                    process_results_dict[vid_name].append(temp_dict)
                except KeyError:
                    process_results_dict[vid_name] = []
                    process_results_dict[vid_name].append(temp_dict)

        # only save specific 100 items
        # visual_data_index = pickle.load(open('/home/xuhaoming/Projects/CVPR2020/utils/visual_data_index.pkl', 'rb'))
        visual_data_index = list(process_results_dict.keys())[:100]
        visual_data = {}
        for idx in visual_data_index:
            visual_data[idx] = process_results_dict[idx]

        # results_folder = './results/Evaluate/Raw_results'
        if save_processed_res != '':
            results_folder = save_processed_res
            baseline_setting = os.path.basename(results_folder).strip().rsplit('_', 2)[0]
            os.makedirs(results_folder, exist_ok=True)
            date = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
            json.dump(visual_data, open(os.path.join(results_folder, f'processed_results_{baseline_setting}.json'), 'w'), indent=4)

        self.processed_results = process_results_dict
        # input_file_name = os.path.basename(raw_results_path).split('.')[0]
        # save_name = os.path.join(os.path.dirname(results_file_path), f'processed_{input_file_name}_no_merge.json')
        # json.dump(process_results_dict, open(save_name, 'w'), indent=4)

        # return process_results_dict

    def compute_IoU_recall_top_n_ours(self, top_n, iou_thresh, nms=False):
        correct_num = 0.0
        total_num = 0.0
        nms_pick_itmes = defaultdict(list)
        # sclips: gt (start, end); iclips: props (start, end); sim_v: match_scores.
        for vid_name, vid_results in self.processed_results.items():
            for query_result in vid_results:
                total_num += 1
                gt = query_result['gt']
                gt_start = gt[0]
                gt_end = gt[1]
                props_predictions = query_result['node_predictions']
                # print gt +" "+str(gt_start)+" "+str(gt_end)
                sim_v = [v[-1] for v in props_predictions]
                # sim_v = [v for v in sentence_image_mat[k]]
                starts = [v[0] for v in props_predictions]
                ends = [v[1] for v in props_predictions]
                # starts = [float(iclip.split("_")[1]) for iclip in iclips]
                # ends = [float(iclip.split("_")[2]) for iclip in iclips]
                if nms:
                    picks = self.nms_temporal(starts, ends, sim_v, iou_thresh - 0.05)
                    raw_levels = {f"level_{i}": item for i, item in enumerate(query_result['level'])}
                    merge_level = []
                    for item in query_result['level']:
                        merge_level.extend(item)
                    merge_level = np.array(merge_level)
                    picks_level = merge_level[picks]
                    temp_new_dict = deepcopy(query_result)
                    temp_new_dict['node_predictions'] = [query_result['node_predictions'][i] for i in picks]
                    temp_new_dict['level'] = picks_level.tolist()
                    nms_pick_itmes[vid_name].append(temp_new_dict)
                else:
                    picks = list(range(len(sim_v)))
                    merge_level = []
                    for item in query_result['level']:
                        merge_level.extend(item)
                    merge_level = np.array(merge_level)
                    picks_level = merge_level[picks]
                    temp_new_dict = deepcopy(query_result)
                    # temp_new_dict['node_predictions'] = [query_result['node_predictions'][i] for i in picks]
                    temp_new_dict['level'] = picks_level.tolist()
                    nms_pick_itmes[vid_name].append(temp_new_dict)
                # sim_argsort=np.argsort(sim_v)[::-1][0:top_n]
                if top_n < len(picks): picks = picks[0:top_n]
                for index in picks:
                    # pred_start = float(iclips[index].split("_")[1])
                    pred_start = props_predictions[index][0]
                    # pred_end = float(iclips[index].split("_")[2])
                    pred_end = props_predictions[index][1]
                    iou = self.calculate_IoU((gt_start, gt_end), (pred_start, pred_end))
                    if iou >= iou_thresh:
                        correct_num += 1
                        break
        # if self.viz_nms:
        self.viz_processed_results = nms_pick_itmes

        # else:
        #     self.viz_processed_results = self.processed_results
        return correct_num, total_num, correct_num / total_num

    def nms_temporal(self, x1, x2, s, overlap):
        pick = []
        assert len(x1) == len(s)
        assert len(x2) == len(s)
        if len(x1) == 0:
            return pick

        union = list(map(operator.sub, x2, x1))  # union = x2-x1
        I = [i[0] for i in sorted(enumerate(s), key=lambda x: x[1])]  # sort and get index

        while len(I) > 0:
            i = I[-1]
            pick.append(i)

            xx1 = [max(x1[i], x1[j]) for j in I[:-1]]
            xx2 = [min(x2[i], x2[j]) for j in I[:-1]]
            inter = [max(0.0, k2 - k1) for k1, k2 in zip(xx1, xx2)]
            o = [inter[u] / (union[i] + union[I[u]] - inter[u]) for u in range(len(I) - 1)]
            I_new = []
            for j in range(len(o)):
                if o[j] <= overlap:
                    I_new.append(I[j])
            I = I_new
        return pick

    def temporal_iou(self, span_A, span_B):
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

    def calculate_IoU(self, i0, i1):
        union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
        inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
        iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
        return iou

    def visualize(self, save_folder):

        data_file = self.viz_processed_results

        frames_dict = json.load(open("/home/xuhaoming/Projects/CVPR2020/data/Charades_frames_info.json", 'r'))
        fps_dict = json.load(open("/home/xuhaoming/Projects/CVPR2020/data/Charades_fps_dict.json", 'r'))
        duration_info = json.load(open("/home/xuhaoming/Projects/cvpr_baseline/One-stage-moment-retrieval/data/Charades/Charades_duration.json", 'r'))
        visual_data_index = pickle.load(open('/home/xuhaoming/Projects/CVPR2020/utils/visual_data_index.pkl', 'rb'))
        visual_data = {}
        for idx in visual_data_index:
            visual_data[idx] = data_file[idx]
        # vid_name_list = list(data_file.keys())[:100]
        # visual_data = {vid: data_file[vid] for vid in vid_name_list}
        # input_file_name = os.path.basename(json_file_path).split('.')[0]
        # json.dump(data_file[:100], open('test_data.json', 'w'))
        for vid_name, vid_info in tqdm(visual_data.items()):
            # fps = float(frames_dict[vid_name]) / float(fps_dict[vid_name])
            fps = float(duration_info[vid_name])
            for query_results in vid_info:
                query = query_results['query'] + '.'
                gt = query_results['gt']
                gt_start, gt_end = gt[0], gt[1]
                node_predictions = query_results['node_predictions']
                levels = query_results['level']
                assert len(levels) == len(node_predictions)
                tags = []
                for i, lev in enumerate(levels):
                    if lev == 0:
                        tags.append(f'First {i}')
                    elif lev == 1:
                        tags.append(f"Second {i}")
                    elif lev == 2:
                        tags.append(f'Third {i}')
                head = list(reversed(tags)) + ['GT']
                props_start = list(reversed(list(map(lambda x: x[0], node_predictions)))) + [gt_start]
                props_end = list(reversed(list(map(lambda x: x[1] - x[0], node_predictions)))) + [gt_end - gt_start]
                scores = list(reversed(list(map(lambda x: "{:.2f}".format(x[2]), node_predictions)))) + [1]
                iou = list(reversed([self.temporal_iou(gt, props) for props in node_predictions])) + [1]
                text = list(map(lambda x: f"s: {float(x[0]):.2f}, iou: {float(x[1]):.2f}", zip(scores, iou)))
                # annotations = [dict(text=f"{x}") for x in scores]
                # assert list(reversed(np.array(node_predictions)[:, 0].tolist())) == props_start[:-1]
                # assert list(reversed(np.array(node_predictions)[:, 1].tolist())) == props_end[:-1]
                fig =  go.Figure()
                # draw gt
                fig.add_trace(go.Bar(
                    y=head,
                    x=props_start,
                    text=[""] * len(head),
                    textposition='inside',
                    textfont=dict(size=30),
                    name='props_start',
                    orientation='h',
                    marker=dict(
                        color='rgba(255, 255, 255, 0.6)',
                        line=dict(color='rgba(255, 255, 255, 1.0)', width=3),
                    ),

                ))
                fig.add_trace(go.Bar(
                    y=head,
                    x=props_end,
                    name='props_end',
                    orientation='h',
                    text=text,
                    textposition='auto',
                    textfont=dict(size=30),

                    marker=dict(
                        color='rgba(66, 218, 202, 0.6)',
                        line=dict(color='rgba(66, 218, 202 , 1.0)', width=3),

                    ),
                ))

                fig.update_layout(barmode='stack',
                                  paper_bgcolor='rgb(255, 255, 255)',
                                  plot_bgcolor='rgb(248, 248, 255)',
                                  showlegend=False,
                                  width=1200, height= 1500,
                                  title=dict(
                                      text=f"vid: {vid_name}, {query} \n" + "GT: {:.2f}  {:.2f}".format(gt_start * fps,
                                                                                                        gt_end * fps),
                                      font=dict(size=30, color="rgb(61, 198, 232)"),
                                      )
                                  ) 
                gt_start = "{:.2f}".format(gt_start * fps)
                gt_end = "{:.2f}".format(gt_end * fps)
                query_tokens = "".join([str(self.word2id[word]) for word in query.strip().replace(".", "").split(' ')])
                fig.write_image(os.path.join(save_folder, f'{vid_name}_{gt_start}_{gt_end}_{query_tokens}.jpg'))

    def run_evaluate(self, iou_topk_dict:dict, do_merge=False, update_score=False, score_weight=1.0, temporal_nms=False, viz_nms=True, do_viz=""):
        assert isinstance(iou_topk_dict, dict)
        self.viz_nms = viz_nms
        ious = iou_topk_dict['iou']
        topks = iou_topk_dict['topk']
        # print('Post processing raw results...')
        if do_merge:
            self._postprocess_raw_results(update_score=update_score, score_weight=score_weight, save_processed_res=do_viz)
        else:
            self._postprocess_raw_results_no_merge(save_processed_res=do_viz)
        # print('Done')
        accuracy_topks = []
        for iou_thresh in ious:
            for topk in topks:
                # print(f'Setting: topk: {topk}, iou: {iou_thresh}, update_score: {update_score}, NMS: {temporal_nms}')
                correct_num_topk, total_num_topk, accuracy_topk = self.compute_IoU_recall_top_n_ours(topk, iou_thresh, temporal_nms)
                # print(f"IoU thresh: {iou_thresh}, NMS: {temporal_nms}, \nR@{topk}: {accuracy_topk}\ncorrect: {correct_num_topk},"
                #       f"  total_num: {total_num_topk}")
                # print(f"\nR@{topk}: {accuracy_topk}\n")
                accuracy_topks.append(accuracy_topk)
        if do_viz:
            print(f'Creating save folder {do_viz}')
            os.makedirs(do_viz, exist_ok=True)
            print(f'Conducting visualization...')
            self.visualize(do_viz)
            print('Done')
        return topks, accuracy_topks




if __name__ == '__main__':
    # raw_results_path = "/home/xuhaoming/Projects/CVPR2020/results/Evaluate/Raw_results/raw_results_2019-07-27-00-33.json"
#     # raw_data = json.load(open(raw_results_path, 'r'))
#     # viz_images_folder = "/home/xuhaoming/Projects/CVPR2020/utils/viz_images/test/"
#     # process_runner = PostProcessRunner(raw_data)
#     # iou_topk_dict = {"iou": [0.5], 'topk': [1, 5]}
#     # process_runner.run_evaluate(do_merge=False, iou_topk_dict=iou_topk_dict)
    results_dict = json.load(open("/home/xuhaoming/Projects/cvpr_baseline/projects/"
                                  "SEED666_baseline6_6_5.6_11_prop32/raw_results_cent_loc.json", 'r'))
    iou_topk_dict = {"iou": [0.5], 'topk': [1, 5]}
    postprocess_runner = PostProcessRunner(results_dict)
    topks, accuracy_topks = postprocess_runner.run_evaluate(iou_topk_dict=iou_topk_dict, temporal_nms=False,
                                                            viz_nms=False,
                                                            do_viz=os.path.join(os.getcwd(),
                                                            f'{os.path.basename(os.getcwd())}_viz_images'))

