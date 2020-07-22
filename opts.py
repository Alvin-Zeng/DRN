import argparse

parser = argparse.ArgumentParser(description="Language Driven Video Temporal Localization implemented in pyTorch")

# =========================== Data Configs ==============================
parser.add_argument('dataset', type=str, default='Charades')
parser.add_argument('--props_file_path', type=str)
parser.add_argument('--feature_type', choices=['C3D', 'I3D', 'MFnet'])
# =========================== Learning Configs ============================
parser.add_argument('--n_epoch', type=int)
parser.add_argument('-b', '--batch_size', type=int)
parser.add_argument('--is_first_stage', action='store_true')
parser.add_argument('--is_second_stage', action='store_true')
parser.add_argument('--is_third_stage', action='store_true')
parser.add_argument('--test_batch_size', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--gpu', type=str)
parser.add_argument('--snapshot_pref', type=str)
parser.add_argument('--glove_weights', type=str)
# parser.add_argument('--normtype', choices=['batch', 'instance'], default='batch')
parser.add_argument('--hidden_dim', type=int)
parser.add_argument('--embedding', type=int)
parser.add_argument('--lstm_layers', type=int)
parser.add_argument('--node_ft_dim', type=int)
parser.add_argument('--graph_num_layers', type=int)
parser.add_argument('--resume', type=str, default="")
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--clip_gradient', type=float)
parser.add_argument('--loss_weights', type=float)
parser.add_argument('--loss_type', choices=['iou', 'bce'])
parser.add_argument('--start_epoch', type=int)
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
parser.add_argument('--weight_decay', '--wd', type=float,
                    metavar='W', help='weight decay (default: 5e-4)')


# =========================== Display Configs ============================
parser.add_argument('--print_freq', type=int)
parser.add_argument('--save_freq', type=int)
parser.add_argument('--eval_freq', type=int)



