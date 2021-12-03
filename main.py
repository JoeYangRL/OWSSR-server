

import logging
import logging.config
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import sys

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch import cuda
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from utils import set_models, set_seed, ModelEMA
import copy
from trainer import train
from parse import set_parser

logger = logging.getLogger(__name__)


"""
# used args paramters
args.world_size: how many gpus you want to use [check]
args.port [check]
args.seed: set random seed [check]
args.eval_only: if only evaluate [check]
args.now_step: current task step (start counting from 1) [check]
args.output_path: path to checkpoint [check]
"""


def main(rank, args):

    ##############################################################################
    # setup
    if args.eval_only:
        log_name = 'step%i_test.log'%(args.now_step)
    else:
        log_name = 'step%i_train.log'%(args.now_step)
    logger.info(f"{log_name}")
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
    log_dict = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'form01': {
                'format': "%(asctime)s - %(levelname)s - %(name)s -   %(message)s", 
                'datefmt': "%m/%d/%Y %H:%M:%S",
            }
        },
        'handlers': {
            'ch': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'form01',
                'stream': 'ext://sys.stderr'
            },
            'fh': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'form01',
                'filename': os.path.join(args.output_path, log_name),
                'mode': 'w'
            }
        },
        'root': {
            'handlers': ['ch', 'fh'],
            'level': 'INFO',
        }
    }
    logging.config.dictConfig(log_dict)
    
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    args.rank = rank
    
    if rank == 0:
        logger.info("---------- start setup -------------------")


    global best_acc
    global best_acc_val
    
    #device = torch.device("cuda")
    device = torch.device("cuda:{}".format(rank) if torch.cuda.is_available() else "cpu" )
    torch.cuda.set_device(device)
    
    if args.world_size > 1:
        torch.distributed.init_process_group(backend='nccl', world_size=args.world_size, rank=rank)
    
    
    args.device = device
    
    if rank == 0:
        logger.info(dict(args._get_kwargs()))
    if args.seed is not None:
        set_seed(args)
    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
        args.writer = SummaryWriter(args.output_path)

    ##############################################################################
    # prepare model
    if rank == 0:
        logger.info("---------- start loading model -------------------")

    model, optimizer, scheduler = set_models(args)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    if args.use_ema:
        ema_model = ModelEMA(args, model, args.ema_decay)
        logger.info("activate ema model")
    else:
        ema_model = None
        logger.info("deactivate ema model")
    
    if args.now_step > 1:
        logger.info("==> Resuming from checkpoint..")
        checkpoint = torch.load(os.path.join(args.output_path, 'model_best_step%i.pth.tar'%(args.now_step-1)))
        best_acc = checkpoint['best_acc']
        model.load_state_dict(torch.load(checkpoint['state_dict'],map_location=device))
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if args.use_ema:
            ema_model.ema.load_state_dict(torch.load(checkpoint['ema_state_dict'],map_location=device))
            old_model = copy.deepcopy(ema_model.ema)
            old_model = old_model.to(args.device)
        else:
            old_model = copy.deepcopy(model)
            old_model = old_model.to(args.device)
        for para in old_model.parameters():
            para.requires_grad = False
        old_model.eval()
        if args.base_task_cls != 0:
            num_cls = int(args.base_task_cls + (args.now_step-1) * ((100-args.base_task_cls) / (args.steps)))
            args.cls_per_step = int((100-args.base_task_cls) / (args.steps))
        else:
            num_cls = int((100/args.steps) * args.now_step)
            args.cls_per_step = int(100/args.steps)
        model.change_output_dim(args, num_cls)
        if args.use_ema:
            ema_model = ModelEMA(args, model, args.ema_decay)
    else:
        old_model = None
    
    #用DDP包装原来的model
    if args.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(args.device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank],
            output_device=rank, find_unused_parameters=True)

    model.zero_grad()
    if not args.eval_only:
        if rank == 0:
            logger.info("---------- enter training process -------------------")
            logger.info(f"  Task = cifar100_B{args.base_task_cls}_{args.steps}steps_{args.num_labeled}num-labeled")
            logger.info(f"  Num Epochs = {args.epoch}")
            logger.info(f"  Batch size per GPU = {args.batch_size}")
            logger.info(f"  Total train batch size = {args.batch_size*args.world_size}")
        train(args, model, old_model, ema_model, optimizer, scheduler, rank)
    """
    #TODO: add eval_only code
    else:
        logger.info("***** Running Evaluation *****")
        logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
        eval_model(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
             ood_loaders, model, ema_model)"""


if __name__ == '__main__':
    #main()
    args = set_parser()
    
    if args.world_size > 1:
        mp.spawn(main, nprocs=args.world_size, args=(args,))
        """processes = []
        for rank in range(args.world_size):
            p = Process(target=main, args=(rank, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()"""
    else:
        main(0, args)