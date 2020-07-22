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
# import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool



class PostProcessRunner:
    def __init__(self, raw_results):
        self.raw_results = raw_results if isinstance(raw_results, dict) else json.load(open(raw_results, 'r'))
        self.word2id = json.load(open("/home/xuhaoming/Projects/cvpr_baseline/"
                                      "projects/SEED666_baseline6_6_5.6_11_prop32/data/word2id.json", 'r'))


    def _postprocess_raw_results(self, save_processed_res):
        process_results_dict = {}
        for vid_name, vid_results in tqdm(self.raw_results.items()):
            for query_result in vid_results:
                query = query_result['query']
                gt = query_result['gt']
                # centerness
                centerness = torch.tensor(query_result['centerness'])[:, None]
                # locations
                locations = torch.tensor(query_result['locations'])[:, None]
                # level
                level_nums = [len(x) for x in query_result['level']]
                levels = list(map(lambda x: torch.tensor(x)[None, :].float(), query_result['level']))
                levels = torch.cat(levels, dim=1).transpose(1,0)
                node_predictions_tensor = torch.tensor(query_result['node_predictions'])
                pred_score_cent_loc = torch.cat((node_predictions_tensor, centerness, locations, levels), dim=1)
                pred_score_cent_loc_list = pred_score_cent_loc.numpy().tolist()
                # sort by scores, index: 2  TODO: make efficient, if there is an api in pytorch
                pred_score_cent_loc = list(sorted(pred_score_cent_loc_list, key=lambda x: x[2], reverse=True))
                pred_score_cent_loc = torch.tensor(pred_score_cent_loc)
                node_predictions = pred_score_cent_loc[:, :3].numpy().tolist()
                centerness = pred_score_cent_loc[:, 3].numpy().tolist()
                locations = pred_score_cent_loc[:, 4].numpy().tolist()
                levels = pred_score_cent_loc[:, 5].numpy().tolist()
                temp_dict = {
                    'query': query,
                    'gt': gt,
                    'node_predictions': node_predictions,
                    # 'level': query_result['level'],  # need to sort
                    'level': [levels, level_nums],
                    'locations': locations,
                    'centerness': centerness
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

        # results_folder = './results/Evaluate/Raw_results'
        if save_processed_res != '':
            results_folder = save_processed_res
            baseline_setting = os.path.basename(results_folder).strip().rsplit('_', 2)[0]
            os.makedirs(results_folder, exist_ok=True)
            date = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
            json.dump(visual_data, open(os.path.join(results_folder, f'processed_results_{baseline_setting}.json'), 'w'), indent=4)

        self.processed_results = process_results_dict


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
                    temp_new_dict = deepcopy(query_result)
                    temp_new_dict['node_predictions'] = [query_result['node_predictions'][i] for i in picks]
                    if self.display_level:
                        merge_level = []
                        for item in query_result['level']:
                            merge_level.extend(item)
                        merge_level = np.array(merge_level)
                        picks_level = merge_level[picks]
                        temp_new_dict['level'] = picks_level.tolist()

                    nms_pick_itmes[vid_name].append(temp_new_dict)
                    del temp_new_dict
                else:
                    picks = list(range(len(sim_v)))
                    temp_new_dict = deepcopy(query_result)
                    if self.display_level:

                        merge_level = []
                        for item in query_result['level']:
                            merge_level.extend(item)
                        merge_level = np.array(merge_level)
                        picks_level = merge_level[picks]
                        temp_new_dict['level'] = picks_level.tolist()
                    nms_pick_itmes[vid_name].append(temp_new_dict)
                    del temp_new_dict

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
        #     if self.display_level:
        #         pass
        #     else:
        #         self.viz_processed_results = self.processed_results




        return correct_num, total_num, correct_num / total_num

    def _draw(self, vid_name, result: dict, save_path):
        '''
        results:{
            'query': string,
            'gt': list,
            'node_prediction': list,
            'level': list
            }
        '''
        query = result['query']
        gt_start, gt_end = result['gt']
        preds = result['node_predictions']
        centerness = result['centerness']
        locations = result['locations']
        if  self.display_level:
            levels = [100] + result['level']  # gt

        num_item = len(preds) + 1 # gt

        # head = list(reversed(tags)) + ['GT']
        props_start = [gt_start] +  list(map(lambda x: x[0], preds))
        props_end = [gt_end] + list(map(lambda x: x[1], preds))
        cent_list = [1] + list(map(lambda x: round(x, 3), centerness))
        loc_list = [""] + list(map(lambda x: round(x, 2), locations))
        scores = [1] + list(map(lambda x: "{:.2f}".format(x[2]), preds))
        iou = [1] + [self.temporal_iou((gt_start, gt_end), props) for props in preds]
        text_list = zip(scores, iou, levels, cent_list, loc_list) if  self.display_level else zip(scores, iou,
                                                                                                  cent_list, loc_list)
        text = list(map(lambda x: f"S: {float(x[0]):.2f} ; IoU: {float(x[1]):.2f} ; level: {x[2]};   C:{x[3]}  ;  L: {x[4]}", text_list)) \
        if self.display_level else list(map(lambda x: f"S: {float(x[0]):.2f}  ;  IoU: {float(x[1]):.2f}  ;  "
        f"C: {x[2]}  ;   L: {x[3]}", text_list))

        # plt.figure(figsize=(12, num_item))
        fig, ax = plt.subplots(figsize=(45, num_item))
        for i in range(num_item):
            ax.add_patch(plt.Rectangle((props_start[i], num_item - i),
                                       props_end[i] - props_start[i], 0.85,
                                       facecolor=(138 / 255, 230 / 255, 223 / 255, 255 / 255),
                                       edgecolor=(66 / 255, 218 / 255, 202 / 255, 255 / 255),
                                       linewidth=1.5))
            ax.text(props_start[i] + 0.03, num_item - i + 0.3, text[i], size=20)
        ax.set_yticks(np.arange(1, num_item+1, 1) + 0.5)
        ax.set_yticklabels(list(reversed(['GT'] + [f'P{i}' for i in range(1, num_item)])))
        plt.yticks(fontsize=25)
        plt.xticks(fontsize=25)
        title = f"#  {vid_name},  {query}. \n GT: {gt_start:.3f}  {gt_end:.3f}"
        plt.title(title, fontdict={'fontsize': 60}, loc='center')
        # ax.set_xticks(np.arange(np.min(props_start), np.max(props_end), 0.1))
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.autoscale()

        plt.savefig(save_path)
        plt.close()

    def _draw_centerness(self, vid_name, result: dict, save_path):
        '''
        results:{
            'query': string,
            'gt': list,
            'node_prediction': list,
            'level': list
            }
        '''
        query = result['query']
        gt_start, gt_end = result['gt']
        preds = result['node_predictions']
        centerness = result['centerness']
        locations = result['locations']
        levels = result['level'][0]
        level_num = result['level'][1]
        if self.display_level:
            levels = [100] + result['level']  # gt

        num_item = len(preds) + 1  # gt

        # head = list(reversed(tags)) + ['GT']
        props_start = [gt_start] + list(map(lambda x: x[0], preds))
        props_end = [gt_end] + list(map(lambda x: x[1], preds))
        cent_list = [1] + list(map(lambda x: round(x, 3), centerness))
        loc_list = [""] + list(map(lambda x: round(x, 2), locations))
        scores = [1] + list(map(lambda x: "{:.2f}".format(x[2]), preds))
        iou = [1] + [self.temporal_iou((gt_start, gt_end), props) for props in preds]
        text_list = zip(scores, iou, levels, cent_list, loc_list) if self.display_level else zip(scores, iou,
                                                                                                 cent_list, loc_list)
        text = list(
            map(lambda x: f"S: {float(x[0]):.2f} ; IoU: {float(x[1]):.2f} ; level: {x[2]};   C:{x[3]}  ;  L: {x[4]}",
                text_list)) \
            if self.display_level else list(map(lambda x: f"S: {float(x[0]):.2f}  ;  IoU: {float(x[1]):.2f}  ;  "
        f"C: {x[2]}  ;   L: {x[3]}", text_list))

        # plt.figure(figsize=(12, num_item))
        # fig, ax = plt.subplots(figsize=(45, num_item))
        # for i in range(num_item):
        #     ax.add_patch(plt.Rectangle((props_start[i], num_item - i),
        #                                props_end[i] - props_start[i], 0.85,
        #                                facecolor=(138 / 255, 230 / 255, 223 / 255, 255 / 255),
        #                                edgecolor=(66 / 255, 218 / 255, 202 / 255, 255 / 255),
        #                                linewidth=1.5))
        #     ax.text(props_start[i] + 0.03, num_item - i + 0.3, text[i], size=20)
        # ax.set_yticks(np.arange(1, num_item + 1, 1) + 0.5)
        # ax.set_yticklabels(list(reversed(['GT'] + [f'P{i}' for i in range(1, num_item)])))
        # plt.yticks(fontsize=25)
        # plt.xticks(fontsize=25)
        title = f"#  {vid_name},  {query}. \n GT: {gt_start:.3f}  {gt_end:.3f}"
        # plt.title(title, fontdict={'fontsize': 60}, loc='center')
        # # ax.set_xticks(np.arange(np.min(props_start), np.max(props_end), 0.1))
        # ax.set_xticks(np.arange(0, 1.1, 0.1))
        # ax.autoscale()
        #
        # plt.savefig(save_path)
        # plt.close()
        x = locations
        y = centerness
        loc_center = list(zip(x, y))
        sorted_loc_center = sorted(loc_center, key=lambda x: x[0])
        sorted_locations, sorted_centerness = zip(*sorted_loc_center)
        # plt.figure(figsize=(12, 12))
        # subplot 1
        # fig, ax = plt.subplot(2, 2, 1)
        fig = plt.figure(figsize=(33, 33))
        ax1 = fig.add_subplot(221)
        # add GT
        gt_start = round(float(gt_start), 3)
        gt_end = round(float(gt_end), 3)
        ax1.add_patch(plt.Rectangle((gt_start, 0.12),
                                    gt_end - gt_start, 0.06,
                                    facecolor=(138 / 255, 230 / 255, 223 / 255, 255 / 255),
                                    edgecolor=(66 / 255, 218 / 255, 202 / 255, 255 / 255),
                                    linewidth=1.5))
        ax1.text(gt_start + 0.01, 0.08, f'{gt_start}', size=20)
        ax1.text(gt_end - 0.06, 0.08, f'{gt_end}', size=20)
        # ax.autoscale()
        plt.plot(sorted_locations, sorted_centerness, label='centerness', color='red', linewidth=2)
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)
        plt.xlabel('Locations', fontsize=30)
        plt.ylabel('Centerness', fontsize=30)
        plt.legend(fontsize=30)

        # subplot 2
        plot2_locations = x[:level_num[0]]
        plot2_centerness = y[:level_num[0]]
        plot2_loc_center = zip(plot2_locations, plot2_centerness)
        sorted_plot2_loc_center = sorted(plot2_loc_center, key=lambda x: x[0])
        sorted_plot2_loc, sorted_plot2_center = zip(*sorted_plot2_loc_center)
        # fig, ax = plt.subplot(2, 2, 2)
        ax2 = fig.add_subplot(222)
        gt_start = round(float(gt_start), 3)
        gt_end = round(float(gt_end), 3)
        ax2.add_patch(plt.Rectangle((gt_start, 0.12),
                                    gt_end - gt_start, 0.06,
                                    facecolor=(138 / 255, 230 / 255, 223 / 255, 255 / 255),
                                    edgecolor=(66 / 255, 218 / 255, 202 / 255, 255 / 255),
                                    linewidth=1.5))
        ax2.text(gt_start + 0.01, 0.08, f'{gt_start}', size=20)
        ax2.text(gt_end - 0.06, 0.08, f'{gt_end}', size=20)
        # ax.autoscale()
        plt.plot(sorted_plot2_loc, sorted_plot2_center, label='Level-1', color='blue', linewidth=2)
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)
        plt.xlabel('Locations', fontsize=30)
        plt.ylabel('Centerness', fontsize=30)
        plt.legend(fontsize=30)

        # subplot 3
        plot3_locations = x[level_num[0]:level_num[1] + level_num[0]]
        plot3_centerness = y[level_num[0]: level_num[1] + level_num[0]]
        plot3_loc_center = zip(plot3_locations, plot3_centerness)
        sorted_plot3_loc_center = sorted(plot3_loc_center, key=lambda x: x[0])
        sorted_plot3_loc, sorted_plot3_center = zip(*sorted_plot3_loc_center)
        # fig, ax = plt.subplot(2, 2, 3)
        ax3 = fig.add_subplot(223)
        gt_start = round(float(gt_start), 3)
        gt_end = round(float(gt_end), 3)
        ax3.add_patch(plt.Rectangle((gt_start, 0.12),
                                    gt_end - gt_start, 0.06,
                                    facecolor=(138 / 255, 230 / 255, 223 / 255, 255 / 255),
                                    edgecolor=(66 / 255, 218 / 255, 202 / 255, 255 / 255),
                                    linewidth=1.5))
        ax3.text(gt_start + 0.01, 0.08, f'{gt_start}', size=20)
        ax3.text(gt_end - 0.06, 0.08, f'{gt_end}', size=20)
        # ax.autoscale()
        plt.plot(sorted_plot3_loc, sorted_plot3_center, label='Level-2', color='green', linewidth=2)
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)
        plt.xlabel('Locations', fontsize=30)
        plt.ylabel('Centerness', fontsize=30)
        plt.legend(fontsize=30)

        # subplot 4
        plot4_locations = x[level_num[1] + level_num[0]:]
        plot4_centerness = y[level_num[1] + level_num[0]:]
        if len(plot4_centerness) != 0:
            plot4_loc_center = zip(plot4_locations, plot4_centerness)
            sorted_plot4_loc_center = sorted(plot4_loc_center, key=lambda x: x[0])
            sorted_plot4_loc, sorted_plot4_center = zip(*sorted_plot4_loc_center)
            # fig, ax = plt.subplot(2, 2, 4)
            ax4 = fig.add_subplot(224)
            gt_start = round(float(gt_start), 3)
            gt_end = round(float(gt_end), 3)
            ax4.add_patch(plt.Rectangle((gt_start, 0.12),
                                        gt_end - gt_start, 0.06,
                                        facecolor=(138 / 255, 230 / 255, 223 / 255, 255 / 255),
                                        edgecolor=(66 / 255, 218 / 255, 202 / 255, 255 / 255),
                                        linewidth=1.5))
            ax4.text(gt_start + 0.01, 0.08, f'{gt_start}', size=20)
            ax4.text(gt_end - 0.06, 0.08, f'{gt_end}', size=20)
            # ax.autoscale()
            plt.plot(sorted_plot4_loc, sorted_plot4_center, label='Level-3', color='mediumslateblue', linewidth=2)
            plt.xlim(0., 1.)
            plt.ylim(0., 1.)
            plt.yticks(fontsize=30)
            plt.xticks(fontsize=30)
            plt.xlabel('Locations', fontsize=30)
            plt.ylabel('Centerness', fontsize=30)
            plt.legend(fontsize=30)

        plt.suptitle(title, fontsize=60, VA='center')
        plt.savefig(save_path)
        plt.close()

    def visualize(self, save_folder):

        data_file = self.viz_processed_results

        duration_info = json.load(open("/home/xuhaoming/Projects/cvpr_baseline/One-stage-moment-retrieval/data/Charades/Charades_duration.json", 'r'))
        visual_data_index = pickle.load(open('/home/xuhaoming/Projects/CVPR2020/utils/visual_data_index.pkl', 'rb'))
        visual_data = {}
        for idx in visual_data_index:
            visual_data[idx] = data_file[idx]

        for vid_name, vid_info in tqdm(visual_data.items()):

            # fps = float(duration_info[vid_name])
            for query_results in vid_info:
                query = query_results['query'] + '.'
                gt = query_results['gt']
                gt_start, gt_end = gt[0], gt[1]

                query_tokens = "".join([str(self.word2id[word]) for word in query.strip().replace(".", "").split(' ')])
                # self._draw(vid_name, query_results, os.path.join(save_folder, f'{vid_name}_{gt_start}_{gt_end}_{query_tokens}.jpg'))
                self._draw_centerness(vid_name, query_results, os.path.join(save_folder, f'{vid_name}_{gt_start}_{gt_end}_{query_tokens}.jpg'))



    def run_evaluate(self, iou_topk_dict:dict, temporal_nms=False, display_level=True, do_viz=""):
        assert isinstance(iou_topk_dict, dict)
        self.display_level = display_level
        ious = iou_topk_dict['iou']
        topks = iou_topk_dict['topk']
        # print('Post processing raw results...')

        self._postprocess_raw_results(save_processed_res=do_viz)

        accuracy_topks = []

        for iou_thresh in ious:
            for topk in topks:
                correct_num_topk, total_num_topk, accuracy_topk = self.compute_IoU_recall_top_n_ours(topk, iou_thresh, temporal_nms)
                accuracy_topks.append(accuracy_topk)
        if do_viz:
            print(f'Creating save folder {do_viz}')
            os.makedirs(do_viz, exist_ok=True)
            print(f'Conducting visualization...')
            self.visualize(do_viz)
            print('Done')
        return topks, accuracy_topks


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


if __name__ == '__main__':
    # raw_results_path = "/home/xuhaoming/Projects/CVPR2020/results/Evaluate/Raw_results/raw_results_2019-07-27-00-33.json"
#     # raw_data = json.load(open(raw_results_path, 'r'))
#     # viz_images_folder = "/home/xuhaoming/Projects/CVPR2020/utils/viz_images/test/"
#     # process_runner = PostProcessRunner(raw_data)
#     # iou_topk_dict = {"iou": [0.5], 'topk': [1, 5]}
#     # process_runner.run_evaluate(do_merge=False, iou_topk_dict=iou_topk_dict)
    results_dict = json.load(open("/home/xuhaoming/Projects/cvpr_baseline/"
                                  "projects/SEED666_baseline6_6_5.6_11_prop32/raw_results_cent_loc.json", 'r'))
    iou_topk_dict = {"iou": [0.5], 'topk': [1, 5]}
    postprocess_runner = PostProcessRunner(results_dict)
    # check the results
    # topks, accuracy_topks = postprocess_runner.run_evaluate(iou_topk_dict=iou_topk_dict, temporal_nms=True)

    topks, accuracy_topks = postprocess_runner.run_evaluate(iou_topk_dict=iou_topk_dict, temporal_nms=False,
                                                            display_level=False,
                                                            do_viz=os.path.join(os.getcwd(),
                                                            f'{os.path.basename(os.getcwd())}_loc_cent_viz_images'))

    print(topks)
    print(accuracy_topks)