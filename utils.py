import random
import numpy as np
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import torch
import torch.optim as optim
import math
from torch.optim.lr_scheduler import LambdaLR
import time
from torch.utils.data import DataLoader
import tqdm
import torch.nn.functional as F
import torch.distributed as dist
import shutil
import logging
from copy import deepcopy
from PIL import Image
"""
# used args paramters
args.seed: 设置随机种子数 [check]
args.wdecay [check]
args.lr [check] 
args.nesterov [check]
args.arch [check]
args.model_depth [check]
args.model_width [check]
args.opt [check]
args.epoch [check]
args.warmup_epoch [check]
args.batch_size [check]
args.num_workers [check]
"""

logger = logging.getLogger(__name__)

def set_seed(args):
    """
    set random seed across the whole process
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.world_size > 0:
        torch.cuda.manual_seed_all(args.seed)

def set_models(args):
    
    model = create_model(args)
    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.opt == 'sgd':
        optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                              momentum=0.9, nesterov=args.nesterov)
    elif args.opt == 'adam':
        optimizer = optim.Adam(grouped_parameters, lr=2e-3)

    # args.epochs = math.ceil(args.total_steps / args.eval_step)
    #scheduler = get_cosine_schedule_with_warmup(
    #    optimizer, args.warmup, args.total_steps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch-args.warmup_epoch, args.lr*0.66)

    return model, optimizer, scheduler

def create_model(args):
    if 'wideresnet' in args.arch:
        import wideresnet as models
        if args.now_step > 1:
            if args.base_task_cls != 0:
                num_cls = args.base_task_cls + (args.now_step-2) * ((100-args.base_task_cls) / (args.steps))
            else:
                num_cls = (100/args.steps) * (args.now_step - 1)
        else:
            if args.base_task_cls != 0:
                num_cls = args.base_task_cls
            else:
                num_cls = 100/args.steps
        
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=0,
                                        num_classes=int(num_cls),
                                        open=True)
    """
    # TODO: add these models if needed
    # model not available right now
    elif args.arch == 'resnext':
        import models.resnext as models
        model = models.build_resnext(cardinality=args.model_cardinality,
                                     depth=args.model_depth,
                                     width=args.model_width,
                                     num_classes=args.num_classes)
    elif args.arch == 'resnet_imagenet':
        import models.resnet_imagenet as models
        model = models.resnet18(num_classes=args.num_classes)"""

    return model

"""
# no need
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)"""

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 1

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 1:
            self.avg = self.sum / (self.count-1)
        else:
            self.avg = 0

def select_memory(args, model, u_train_dataset, memory_dataset):

    """
    only select memory from unlabeled data before current training is well initiailized
    """
    data_time = AverageMeter()
    end = time.time()
    u_train_dataset.init_index()

    ##############################################################################
    # sample memory

    test_loader = DataLoader(
        u_train_dataset,
        batch_size=(args.batch_size*args.mu_u*2),
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch_idx, ((_, _, inputs), targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            #logger.info(f"{inputs.size(0)}")
            outputs, outputs_open = model(inputs)
            #logger.info(f"outputs: {outputs.shape}, outputs_open: {outputs_open.shape}")
            outputs = F.softmax(outputs, 1)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.range(0, out_open.size(0) - 1).long().cuda()
            pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0, pred_close]
            known_ind = unk_score < 0.5
            if batch_idx == 0:
                known_all = known_ind
            else:
                known_all = torch.cat([known_all, known_ind], 0)
    known_all = known_all.data.cpu().numpy()
    memory_idx = np.where(known_all != 0)[0]
    u_train_idx = np.where(known_all == 0)[0]
    print("memory selected ratio %s"%( (len(memory_idx)/ len(known_all))))
    print("memory index selected:", len(memory_idx))
    print("total unlabeled data: ", len(known_all))
    memory_dataset.select_idx(memory_idx)
    #u_train_dataset.select_idx(u_train_idx)

def split_u_dataset(args, model, u_train_dataset, memory_dataset, confirm_dataset):

    """
    split the whole u_train_dataset into memory/confirm_known/still unknown
    """
    
    data_time = AverageMeter()
    end = time.time()
    u_train_dataset.init_index()

    ##############################################################################
    # sample memory & confirm

    test_loader = DataLoader(
        u_train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch_idx, ((_, _, inputs), targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            outputs, outputs_open = model(inputs)
            outputs = F.softmax(outputs, 1)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.range(0, out_open.size(0) - 1).long().cuda()
            pred_close = outputs.data.max(1)[1] # [bsz] output: cls_id with max value
            old_cls_list = pred_close < (model.num_classes-args.cls_per_step)
            new_cls_list = pred_close >= (model.num_classes-args.cls_per_step)
            unk_score = out_open[tmp_range, 0, pred_close]  
            known_ind = unk_score < 0.5 # True/False list
            if batch_idx == 0:
                confirm = (known_ind * new_cls_list)
                memory = (known_ind * old_cls_list)
            else:
                confirm = torch.cat([confirm, known_ind * new_cls_list], 0)
                memory = torch.cat([memory, known_ind * old_cls_list], 0)
    confirm = confirm.data.cpu().numpy()
    memory = memory.data.cpu().numpy()

    memory_idx = np.where(memory != 0)[0] # id
    confirm_idx = np.where(confirm != 0)[0]
    all_idx = np.array([i for i in range(len(memory))])
    u_train_idx = np.setdiff1d(all_idx, memory_idx)
    #u_train_idx = np.setdiff1d(u_train_idx, confirm_idx)

    if args.rank == 0:
        logger.info("memory selected ratio %s"%((len(memory_idx)/ len(memory))))
        logger.info("memory index selected:%s"%len(memory_idx))
        logger.info("confirm selected ratio %s"%((len(confirm_idx)/ len(memory))))
        logger.info("confirm index selected:%s"%len(confirm_idx))
        logger.info("rest unselected ratio %s"%((len(u_train_idx)/ len(memory))))
        logger.info("rest unselected:%s"%len(u_train_idx))
        logger.info("total unlabeled data:%s"%len(memory))
    model.train()
    memory_dataset.select_idx(memory_idx)
    confirm_dataset.select_idx(confirm_idx)
    #u_train_dataset.select_idx(u_train_idx)

def select_confirm(args, model, u_train_dataset, confirm_dataset, memory_dataset=None):

    """
    select confirmed samples, used in task step 1
    """
    data_time = AverageMeter()
    end = time.time()
    confirm_dataset.init_index()

    ##############################################################################
    # sample memory

    test_loader = DataLoader(
        confirm_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch_idx, ((inputs, _, _), targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            outputs, outputs_open = model(inputs)
            outputs = F.softmax(outputs, 1)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.range(0, out_open.size(0) - 1).long().to(args.device)
            pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0, pred_close]
            known_ind = unk_score < 0.5
            if batch_idx == 0:
                known_all = known_ind
            else:
                known_all = torch.cat([known_all, known_ind], 0)
    known_all = known_all.data.cpu().numpy()
    confirm_idx = np.where(known_all != 0)[0]
    u_train_idx = np.where(known_all == 0)[0]
    if memory_dataset is not None:
        confirm_idx = np.setdiff1d(confirm_idx, memory_dataset.idx)
        u_train_idx = np.setdiff1d(u_train_idx, memory_dataset.idx)
    
    if args.rank == 0:
        if memory_dataset != None:
            logger.info("memory selected ratio %s"%(len(memory_dataset.idx) / len(known_all)))
            logger.info("memory index selected:%s"%len(memory_dataset.idx))
        logger.info("confirm selected ratio %s"%(len(confirm_idx)/ len(known_all)))
        logger.info("confirm index selected:%s"%len(confirm_idx))
        logger.info("rest unselected ratio %s"%(len(u_train_idx)/ len(known_all)))
        logger.info("rest unselected:%s"%len(u_train_idx))
        logger.info("total unlabeled data:%s"%len(known_all))
    model.train()
    confirm_dataset.select_idx(confirm_idx)
    #u_train_dataset.select_idx(u_train_idx)

def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt

def save_checkpoint(state, is_best, checkpoint, now_step, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best_step%i.pth.tar'%(now_step)))

class ModelEMA(object):
    def __init__(self, args, model, decay):
        self.ema = deepcopy(model)
        self.ema.to(args.device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])

def load_image(path):

    return Image.open(path).convert('RGB')