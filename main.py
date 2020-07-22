import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from utils.Recorder import Recorder
import numpy as np
import time
import shutil
from utils.detection_metrics import segment_tiou
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from opts import parser
from utils.tools import AverageMeter, Prepare_logger, get_and_save_args
import json
import pickle
import os
import random
from model.main_model import mainModel
# from VGG_dataset import CharadesSTA, collate_data
from dataset import CharadesSTA, collate_data
# from TACoS_dataset import TACoS, collate_data
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
from utils.evaluate_utils import PostProcessRunner


best_top1 = 0
best_top5 = 0
best_top1_top5 = 0
best_top5_top1 = 0
best_top1_epoch = 0
best_top5_epoch = 0
SEED = 222
#random.seed(SEED)
#torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)
#np.random.seed(SEED)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


def main():
    global args, logger, writer, dataset_configs
    global best_top1_epoch, best_top5_epoch, best_top1, best_top5, best_top1_top5, best_top5_top1
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()

    # ================== GPU setting ===============
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    """copy codes and creat dir for saving models and logs"""
    if not os.path.isdir(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    logger = Prepare_logger(args)
    logger.info('\ncreating folder: ' + args.snapshot_pref)

    if not args.evaluate:
        writer = SummaryWriter(args.snapshot_pref)
        recorder = Recorder(args.snapshot_pref)
        recorder.writeopt(args)

    logger.info('\nruntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))

    """prepare dataset and model"""
    # word2idx = json.load(open('./data/dataset/TACoS/TACoS_word2id_glove_lower.json', 'r'))
    # train_dataset = TACoS(args, split='train')
    # test_dataset = TACoS(args, split='test')
    word2idx = json.load(open('./data/dataset/Charades/Charades_word2id.json', 'r'))
    train_dataset = CharadesSTA(args, split='train')
    test_dataset = CharadesSTA(args, split='test')
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_data, num_workers=8, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size,
        shuffle=False, collate_fn=collate_data, num_workers=8, pin_memory=True
    )
    vocab_size = len(word2idx)

    lr = args.lr
    n_epoch = args.n_epoch

    main_model = mainModel(vocab_size, args, hidden_dim=512, embed_dim=300,
                           bidirection=True, graph_node_features=1024)

    if os.path.exists(args.glove_weights):
        logger.info("Loading glove weights")
        main_model.query_encoder.embedding.weight.data.copy_(torch.load(args.glove_weights))
    else:
        logger.info("Generating glove weights")
        main_model.query_encoder.embedding.weight.data.copy_(glove_init(word2idx))

    main_model = nn.DataParallel(main_model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            pretrained_dict = checkpoint['state_dict']
            # only resume part of model paramete
            model_dict = main_model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            main_model.load_state_dict(model_dict)
            # main_model.load_state_dict(checkpoint['state_dict'])
            logger.info(("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.evaluate, checkpoint['epoch'])))
        else:
            logger.info(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.evaluate:
        topks, accuracy_topks = evaluate(main_model, test_dataloader, word2idx, False)
        for ind, topk in enumerate(topks):
            print("R@{}: {:.1f}\n".format(topk, accuracy_topks[ind] * 100))
        return

    learned_params = None
    if args.is_first_stage:
        for name, value in main_model.named_parameters():
            if 'iou_scores' in name or 'mix_fc' in name:
                value.requires_grad = False
        learned_params = filter(lambda p: p.requires_grad, main_model.parameters())
        n_epoch = 10
    elif args.is_second_stage:
        head_params = main_model.module.fcos.head.iou_scores.parameters()
        fc_params = main_model.module.fcos.head.mix_fc.parameters()
        learned_params = list(head_params) + list(fc_params)
        lr /= 100
    elif args.is_third_stage:
        learned_params = main_model.parameters()
        lr /= 10000

    optimizer = torch.optim.Adam(learned_params, lr)

    for epoch in range(args.start_epoch, n_epoch):

        train_loss = train_epoch(main_model, train_dataloader, optimizer, epoch)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.n_epoch - 1:

            val_loss, topks, accuracy_topks = validate_epoch(
                main_model, test_dataloader, epoch, word2idx, False
            )

            for ind, topk in enumerate(topks):
                writer.add_scalar('test_result/Recall@top{}'.format(topk), accuracy_topks[ind]*100, epoch)

            is_best_top1 = (accuracy_topks[0]*100) > best_top1
            best_top1 = max((accuracy_topks[0]*100), best_top1)
            if is_best_top1:
                best_top1_epoch = epoch
                best_top1_top5 = accuracy_topks[1]*100
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': main_model.state_dict(),
                'loss': val_loss,
                'top1': accuracy_topks[0]*100,
                'top5': accuracy_topks[1]*100,
            }, is_best_top1, epoch=epoch, top1=accuracy_topks[0]*100, top5=accuracy_topks[1]*100)

            is_best_top5 = (accuracy_topks[1]*100) > best_top5
            best_top5= max((accuracy_topks[1]*100), best_top5)
            if is_best_top5:
                best_top5_epoch = epoch
                best_top5_top1= accuracy_topks[0] * 100
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': main_model.state_dict(),
                'loss': val_loss,
                'top1': accuracy_topks[0]*100,
                'top5': accuracy_topks[1]*100,
            }, is_best_top5, epoch=epoch, top1=accuracy_topks[0]*100, top5=accuracy_topks[1]*100)

            writer.add_scalar('test_result/Best_Recall@top1', best_top1, epoch)
            writer.add_scalar('test_result/Best_Recall@top5', best_top5, epoch)

            logger.info(
                "R@1: {:.2f}, R@5: {:.2f}, epoch: {}\n".format(
                    accuracy_topks[0] * 100, accuracy_topks[1] * 100, epoch)
            )
            logger.info(
                "Current best top1: R@1: {:.2f}, R@5: {:.2f}, epoch: {} \n".format(
                    best_top1, best_top1_top5, best_top1_epoch)
            )
            logger.info(
                "Current best top5: R@1: {:.2f}, R@5: {:.2f}, epoch: {} \n".format(
                    best_top5_top1, best_top5, best_top5_epoch)
            )


def train_epoch(model, train_dataloader, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    iou_losses = AverageMeter()
    center_losses = AverageMeter()
    inner_losses = AverageMeter()
    end = time.time()

    model.train()
    optimizer.zero_grad()

    for iter, (vid_names, props_start_end, props_features,
               gt_start_end, query_tokens, query_len, props_num, num_frames) in enumerate(train_dataloader):

        data_time.update(time.time() - end)
        bs = props_features.size(0)

        box_lists, loss_dict = model(
            query_tokens, query_len, props_features, props_start_end, gt_start_end, props_num, num_frames
        )

        if args.is_second_stage:
            loss = loss_dict['loss_iou']
        else:
            loss = sum(loss for loss in loss_dict.values())

        losses.update(loss.item(), bs)
        cls_losses.update(loss_dict["loss_cls"].item(), bs)
        reg_losses.update(loss_dict["loss_reg"].item(), bs)
        iou_losses.update(loss_dict['loss_iou'].item(), bs)
        # center_losses.update(loss_dict["loss_centerness"].item(), bs)
        # inner_losses.update(loss_dict['loss_innerness'].item(), bs)

        # print(losses.avg)
        if loss != 0:
            loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
            #     logger.info("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        writer.add_scalar('train_data/loss', losses.val, epoch * len(train_dataloader) + iter + 1)
        writer.add_scalar('train_data/cls_loss', cls_losses.val, epoch * len(train_dataloader) + iter + 1)
        writer.add_scalar('train_data/reg_loss', reg_losses.val, epoch * len(train_dataloader) + iter + 1)
        writer.add_scalar('train_data/iou_loss', iou_losses.val, epoch * len(train_dataloader) + iter + 1)
        # writer.add_scalar('train_data/center_loss', center_losses.val, epoch * len(train_dataloader) + iter + 1)
        # writer.add_scalar('train_data/inner_loss', inner_losses.val, epoch * len(train_dataloader) + iter + 1)

        if iter % args.print_freq == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, iter, len(train_dataloader), batch_time=batch_time,data_time=data_time, loss=losses)
            )

    writer.add_scalar('train_epoch_data/epoch_loss', losses.avg, epoch)
    writer.add_scalar('train_epoch_data/epoch_cls_loss', cls_losses.avg, epoch)
    writer.add_scalar('train_epoch_data/epoch_reg_loss', reg_losses.avg, epoch)
    writer.add_scalar('train_epoch_data/epoch_iou_loss', iou_losses.avg, epoch)
    # writer.add_scalar('train_epoch_data/epoch_center_loss', center_losses.avg, epoch )
    # writer.add_scalar('train_epoch_data/epoch_inner_loss', inner_losses.avg, epoch)

    return losses.avg


def validate_epoch(trained_model, test_dataloader, epoch, word2idx, save_results=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    center_losses = AverageMeter()
    iou_losses = AverageMeter()
    end = time.time()

    trained_model.eval()
    results_dict = {}
    id2word = {idx: word for word, idx in word2idx.items()}

    with torch.no_grad():
        for iter, (vid_names, props_start_end, props_features,
                   gt_start_end, query_tokens, query_len, props_num, num_frames) in enumerate(test_dataloader):

            data_time.update(time.time() - end)
            bs = props_features.size(0)

            box_lists, loss_dict = trained_model(
                query_tokens, query_len, props_features, props_start_end, gt_start_end, props_num, num_frames
            )

            if args.is_second_stage:
                loss = loss_dict['loss_iou']
            else:
                loss = sum(loss for loss in loss_dict.values())

            losses.update(loss.item(), bs)
            cls_losses.update(loss_dict["loss_cls"].item(), bs)
            reg_losses.update(loss_dict["loss_reg"].item(), bs)
            iou_losses.update(loss_dict['loss_iou'].item(), bs)
            # center_losses.update(loss_dict["loss_centerness"].item(), bs)
            # inner_losses.update(loss_dict['loss_innerness'].item(), bs)

            batch_time.update(time.time() - end)
            end = time.time()

            if iter % args.print_freq == 0:
                logger.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch, iter, len(test_dataloader), batch_time=batch_time, data_time=data_time, loss=losses)
                )

            for i in range(bs):
                vid_name = vid_names[i]
                query_length = query_len[i]
                query = (' ').join(list(map(lambda x: id2word[x.item()], query_tokens[i, :query_length])))
                gt = gt_start_end[i].numpy().tolist()
                valid_props_num = props_num[i]

                per_vid_detections = box_lists[i]["detections"]
                per_vid_scores = box_lists[i]["scores"]
                per_vid_level = box_lists[i]['level']

                props_pred = torch.cat((per_vid_detections, per_vid_scores.unsqueeze(-1)), dim=-1)
                # edge_pred_info = edge_pred[i, :valid_props_num, :valid_props_num, :].permute(1, 2, 0).contiguous()
                temp_dict = {
                    'query': query,
                    'gt': gt,
                    'node_predictions': props_pred.cpu().numpy().tolist(),
                    'edge_predictions': props_pred.cpu().numpy().tolist(),
                    'level': per_vid_level
                }
                try:
                    results_dict[vid_name].append(temp_dict)
                except KeyError:
                    results_dict[vid_name] = []
                    results_dict[vid_name].append(temp_dict)

        writer.add_scalar('val_epoch_data/epoch_loss', losses.avg, epoch)
        writer.add_scalar('val_epoch_data/epoch_cls_loss', cls_losses.avg, epoch)
        writer.add_scalar('val_epoch_data/epoch_reg_loss', reg_losses.avg, epoch)
        writer.add_scalar('val_epoch_data/epoch_iou_loss', iou_losses.avg, epoch)
        # writer.add_scalar('val_epoch_data/epoch_center_loss', center_losses.avg, epoch)
        # writer.add_scalar('val_epoch_data/epoch_inner_loss', inner_losses.avg, epoch)

        if save_results:
            results_folder = './results/Evaluate/Raw_results'
            os.makedirs(results_folder, exist_ok=True)
            date = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
            json.dump(results_dict, open(os.path.join(results_folder, f'raw_results_{date}.json'), 'w'), indent=4)
        iou_topk_dict = {"iou": [0.5], 'topk': [1, 5]}
        postprocess_runner = PostProcessRunner(results_dict)
        topks, accuracy_topks = postprocess_runner.run_evaluate(iou_topk_dict=iou_topk_dict, temporal_nms=True)

    return losses.avg, topks, accuracy_topks


def save_checkpoint(state, is_best, epoch, top1, top5):
    if is_best:
        best_name = "{}/model_{}_epoch{}_top1_{:.3f}_top5_{:.3f}_model_best.pth.tar".format(
            args.snapshot_pref, args.dataset, epoch, top1, top5)
        torch.save(state, best_name)


def glove_init(word2id):
    glove = list(open('/home/alvin/Projects/Lang-Driven-Temp-Localization/data/glove.6B.300d.txt', 'r'))
    full_glove = {}
    for line in glove:
        values = line.strip('\n').split(" ")
        word = values[0]
        vector = np.asarray([float(e) for e in values[1:]])
        full_glove[word] = vector
    full_glove_keys = list(full_glove.keys())
    weight = torch.zeros((len(word2id)+1, 300), dtype=torch.float)
    for word, idx in word2id.items():
        if word in list(full_glove_keys):
            glove_vector = torch.from_numpy(full_glove[word])
            weight[idx] = glove_vector
    torch.save(weight, args.glove_weights)

    return weight


def evaluate(trained_model, test_dataloader, word2idx, save_results=True):
    trained_model.eval()
    results_dict = {}
    id2word = {idx: word for word, idx in word2idx.items()}

    with torch.no_grad():
        for iterations, (vid_names, props_start_end, props_features,
                         gt_start_end, query_tokens, query_len, props_num, num_frames) in enumerate(tqdm(test_dataloader)):

            start = time.time()
            bs = props_features.size(0)

            box_lists, loss_dict = trained_model(
                query_tokens, query_len, props_features, props_start_end, gt_start_end, props_num, num_frames
            )

            for i in range(bs):
                vid_name = vid_names[i]
                query_length = query_len[i]
                query = (' ').join(list(map(lambda x: id2word[x.item()], query_tokens[i, :query_length])))
                gt = gt_start_end[i].numpy().tolist()
                valid_props_num = props_num[i]

                per_vid_detections = box_lists[i]["detections"]
                per_vid_scores = box_lists[i]["scores"]
                per_vid_level = box_lists[i]['level']
                per_vid_locations = box_lists[i]['locations']

                # per_vid_centerness = box_lists[i]['centerness']
                props_pred = torch.cat((per_vid_detections, per_vid_scores.unsqueeze(-1)), dim=-1)
                # edge_pred_info = edge_pred[i, :valid_props_num, :valid_props_num, :].permute(1, 2, 0).contiguous()
                temp_dict = {
                    'query': query,
                    'gt': gt,
                    'node_predictions': props_pred.cpu().numpy().tolist(),
                    'edge_predictions': props_pred.cpu().numpy().tolist(),
                    'level': per_vid_level,
                    # 'centerness': per_vid_centerness.cpu().numpy().tolist(),
                    'locations': per_vid_locations.cpu().numpy().tolist()
                }
                try:
                    results_dict[vid_name].append(temp_dict)
                except KeyError:
                    results_dict[vid_name] = []
                    results_dict[vid_name].append(temp_dict)

                batch_time = time.time() - start
                # logger.info(f"Query: [{i}/{iterations}/{len(test_dataloader)}]\t" +
                #             "Time {:.3f}\t".format(batch_time))

        if save_results:
            results_folder = './results/Evaluate/Raw_results'
            os.makedirs(results_folder, exist_ok=True)
            date = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
            json.dump(results_dict, open(os.path.join(results_folder, f'raw_results_{date}.json'), 'w'), indent=4)
        iou_topk_dict = {"iou": [0.5], 'topk': [1, 5]}
        postprocess_runner = PostProcessRunner(results_dict)
        topks, accuracy_topks = postprocess_runner.run_evaluate(
            iou_topk_dict=iou_topk_dict, temporal_nms=True,
        )

    return topks, accuracy_topks


def compute_loss(node_pred, node_label, edge_pred, edge_label, props_num, criterion):

    node_pred = node_pred.squeeze(-1)
    edge_pred = edge_pred.permute(0, 2, 3, 1)[:, :, :, 0].squeeze(-1)

    sorted_props_num, sorted_indices = torch.sort(props_num, descending=True)
    sorted_node_pred = node_pred[sorted_indices]
    sorted_node_label = node_label[sorted_indices]
    sorted_edge_pred = edge_pred[sorted_indices]
    sorted_edge_label = edge_label[sorted_indices]

    node_pred, _ = pack_padded_sequence(sorted_node_pred, sorted_props_num, batch_first=True)
    node_label, _ = pack_padded_sequence(sorted_node_label, sorted_props_num, batch_first=True)
    edge_pred, _ = pack_padded_sequence(sorted_edge_pred, sorted_props_num, batch_first=True)
    edge_label, _ = pack_padded_sequence(sorted_edge_label, sorted_props_num, batch_first=True)

    node_loss = criterion(node_pred, node_label.float())
    edge_loss = criterion(edge_pred, edge_label.float())

    return node_loss, edge_loss


def additional_loss(node_pred, props_s_e, gt_s_e, criterion, props_num):

    node_pred = node_pred.squeeze(-1)
    iou_list =[segment_tiou(gt_s_e[n_bs].unsqueeze(0), props_s_e[n_bs]) for n_bs in range(node_pred.size(0))]
    iou_result = torch.cat(iou_list, 0)
    _, indices = iou_result.max(dim=1)
    node_label = indices.cuda()
    bs = node_pred.size(0)
    total_loss = 0.0
    for i in range(bs):
        try:
            total_loss += criterion(node_pred[i, :props_num[i]].unsqueeze(0), node_label[i].unsqueeze(0)).item()
        except:
            a = 0
    # node_label[torch.arange(indices.size(0)), indices] = 1.0
    # node_loss = criterion(node_pred, node_label)
    total_loss /= bs
    # return node_loss
    return total_loss


if __name__ == '__main__':
    main()
