import plotly.graph_objects as go
import json
import os
from tqdm import tqdm
import numpy as np
import pickle
import json

def visualize(json_file_path:str):
    data_file = json.load(open(json_file_path, 'r'))

    frames_dict = json.load(open("/home/xuhaoming/Projects/CVPR2020/data/Charades_frames_info.json", 'r'))
    fps_dict = json.load(open("/home/xuhaoming/Projects/CVPR2020/data/Charades_fps_dict.json", 'r'))
    # visual_data_index = pickle.load(open('./visual_data_index.pkl', 'rb'))
    # visual_data = {}
    # for idx in visual_data_index:
    #     visual_data[idx] = data_file[idx]
    vid_name_list = list(data_file.keys())[:100]
    visual_data = {vid: data_file[vid] for vid in vid_name_list}
    input_file_name = os.path.basename(json_file_path).split('.')[0]
    # json.dump(data_file[:100], open('test_data.json', 'w'))
    for vid_name, vid_info in tqdm(visual_data.items()):
        fps = float(frames_dict[vid_name]) / float(fps_dict[vid_name])
        total_gt = [query['gt'] for query in vid_info]
        for query_results in vid_info:
            query = query_results['query']+'.'
            gt = query_results['gt']
            gt_start, gt_end = gt[0], gt[1]
            other_gt = list(filter(lambda x: x != gt, total_gt))
            node_predictions = query_results['node_predictions']
            head = list(reversed([f"p {i}" for i in range(len(node_predictions))])) + [f'GT{i}'for i in range(len(total_gt))]
            props_start = list(reversed(list(map(lambda x: x[0], node_predictions)))) + [gt_start] + [x[0] for x in other_gt]
            props_end = list(reversed(list(map(lambda x: x[1] - x[0], node_predictions)))) + [gt_end - gt_start] + [x[1]-x[0] for x in other_gt]
            scores = list(reversed(list(map(lambda x: "{:.4f}".format(x[2]), node_predictions)))) + [1] * len(total_gt)
            iou = list(reversed([temporal_iou(gt, props) for props in node_predictions])) + [1] * len(total_gt)
            text = list(map(lambda x: f"s: {x[0]}, iou: {x[1]}", zip(scores, iou)))
            # annotations = [dict(text=f"{x}") for x in scores]
            # assert list(reversed(np.array(node_predictions)[:, 0].tolist())) == props_start[:-1]
            # assert list(reversed(np.array(node_predictions)[:, 1].tolist())) == props_end[:-1]
            fig = go.Figure()
            gt_full_colors = ['#8d4bbb', '#f9906f', '#ef7a82', '#b0a4e3', '#815476', '#cca4e3', '#4b5cc4',
                              '#425066', '#a98175']
            colors = ['rgba(66, 218, 202, 0.6)'] * 32
            colors[-1] = 'crimson'
            colors += gt_full_colors[:len(other_gt)]
            line_colors = ['rgba(66, 218, 202 , 1.0)'] * 32
            line_colors[-1] = "crimson"
            line_colors += gt_full_colors[:len(other_gt)]
            # draw gt
            fig.add_trace(go.Bar(
                y=head,
                x=props_start,
                text=[""] * len(head),
                textposition='inside',
                textfont=dict(size=15),
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
                textfont=dict(size=18),

                marker=dict(
                    # color='rgba(66, 218, 202, 0.6)',
                    color=colors,
                    line=dict(color=line_colors, width=3),
                ),
            )),


            fig.update_layout(barmode='stack',
                              paper_bgcolor='rgb(255, 255, 255)',
                              plot_bgcolor='rgb(248, 248, 255)',
                              showlegend=False,
                              width=1200, height=1000,
                              title=dict(text=f"vid: {vid_name}, {query} \n" + "GT: {:.2f}  {:.2f}".format(gt_start * fps, gt_end * fps),
                                         font=dict(size=30, color="rgb(61, 198, 232)"),
                                         )
                              )

            save_folder = f"./viz_images_test/{input_file_name}/"
            os.makedirs(save_folder, exist_ok=True)

            fig.write_image(os.path.join(save_folder, f'{vid_name}_{gt_start}_{gt_end}.jpg'))


def sample_visual_data(input_file_path:str, num=100):
    data_file = json.load(open(input_file_path, 'r'))
    vid_name_list = list(data_file.keys())
    visual_vid_index = vid_name_list[:num]
    pickle.dump(visual_vid_index, open('./visual_data_index.pkl', 'wb'))


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
        iou = float(inter[1] - inter[0]) / float(union[1] - union[0])
        iou_str = "{:.3f}".format(iou)
        return iou_str


if __name__ == '__main__':
    json_file_path1 = "/home/xuhaoming/Projects/CVPR2020/results/Evaluate/Raw_results/processed_raw_results_2019-07-25-18-14_no_merge.json"
    # json_file_path2 = "/home/xuhaoming/Projects/CVPR2020/results/Evaluate/Raw_results/processed_raw_results_Glove_init_no_merge.json"
    # json_file_path = "/home/xuhaoming/Projects/CVPR2020/results/Evaluate/Raw_results/test_data.json"
    visualize(json_file_path1)
    # visualize(json_file_path2)