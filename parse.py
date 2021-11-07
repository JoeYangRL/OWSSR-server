import argparse

__all__ = ['set_parser']

def set_parser():
    parser = argparse.ArgumentParser(description='PyTorch OWSSR Training')
    ## Computational Configurations

    parser.add_argument('--base_task_cls', default=0, type=int,
                        help='base task cls number')
    parser.add_argument('--steps', default=5, type=int,
                        help='incremental steps')
    parser.add_argument('--save_root', default='./data', type=str,
                        help='root path to save datasets')
    parser.add_argument('--root', default='/data/yangruilin', type=str,
                        help='root path of original datasets')
    parser.add_argument('--world_size', default=4, type=int,
                        help='how many gpus you want to use')
    parser.add_argument('--port', default='29500', type=str,
                        help='port for multi-gpu communication')
    parser.add_argument('--output_path', default='./output', type=str,
                        help='path to checkpoint you want to load')
    parser.add_argument('--seed', default=1993, type=int,
                        help='set random seed')
    parser.add_argument('--eval_only', default=0, type=int,
                        help='> 0 to open eval_only mode')
    parser.add_argument('--now_step', default=1, type=int,
                        help='current task step (start counting from 1)')
    parser.add_argument('--online_distill', default=1, type=int,
                        help='0 for offline distill mode')
    parser.add_argument('--mu_u', default=2, type=int,
                        help='mu for u_bsz')
    parser.add_argument('--mu_c', default=2, type=int,
                        help='mu for c_bsz')
    parser.add_argument('--mu_m', default=2, type=int,
                        help='mu for m_bsz')
    parser.add_argument('--lambda_oem', default=0.1, type=float,
                        help='weight for open enrtopy min loss')
    parser.add_argument('--lambda_socr', default=0.5, type=float,
                        help='weight for open consistency regularization loss')
    parser.add_argument('--lambda_dist', default=1.0, type=float,
                        help='weight for distillation loss')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--lr', default=0.03, type=float,
                        help='learning rate')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--arch', default='wideresnet', type=str,
                        help='backbone arch')
    parser.add_argument('--model_depth', default=28, type=int,
                        help='model depth for wideresnet')
    parser.add_argument('--model_width', default=2, type=int,
                        help='model width for wideresnet')
    parser.add_argument('--opt', default='sgd', type=str,
                        choices=['sgd', 'adam'],
                        help='optimize name')
    parser.add_argument('--epoch', default=300, type=int,
                        help='training epoch')
    parser.add_argument('--warmup_epoch', default=10, type=int,
                        help='warmup_epoch')
    parser.add_argument('--start_fix', default=20, type=int,
                        help='epoch before start to add fixmatch')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch_size for labeled data')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='num_workers for dataloader')
    parser.add_argument('--T', default=2.0, type=float,
                        help='distillation temperature')
    parser.add_argument('--num_labeled', default=50, type=int,
                        help='labeled sample for each class')
    parser.add_argument('--expand_labels', action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--num_expand_x', default=100, type=int,
                        help='expand labeled dataset for how many times')
    parser.add_argument('--memory_size', default=2000, type=int,
                        help='memory_size')
    parser.add_argument('--unlabeled_mu', default=20, type=int,
                        help='ratio unlabeled data/labeled data')
    parser.add_argument('--imagenet_valid_size', default=1000, type=int,
                        help='imagenet_valid_size')
    parser.add_argument('--timestamp', default='', type=str,
                        help='time when created the using dataset')
    parser.add_argument('--eval_step', default=512, type=int,
                        help='iters per epoch')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='threshold to choose confident samples in FixMatch')
    
    args = parser.parse_args()
    return args