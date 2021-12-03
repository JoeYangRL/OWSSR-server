import logging
import time
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import roc_auc_score
from utils import AverageMeter

logger = logging.getLogger(__name__)

def test(args, test_loader, model, val=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    acc = AverageMeter()
    unk = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    test_loader = tqdm(test_loader,
                        #disable=args.local_rank not in [-1, 0]
                        disable=False)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device) # [bsz, 3, x, y]
            #logger.info(f"inputs: {inputs.shape}")
            targets = targets.to(args.device) # [bsz]
            #logger.info(f"targets: {targets.shape}")
            outputs, outputs_open = model(inputs)
            outputs = F.softmax(outputs, 1) # [bsz, num_known_cls]
            #logger.info(f"outputs: {outputs.shape}")
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1) # [bsz, 2, num_known_cls]
            #logger.info(f"out_open: {out_open.shape}")
            tmp_range = torch.range(0, out_open.size(0) - 1).long().cuda() # [bsz]
            #logger.info(f"tmp_range: {tmp_range.shape}")
            pred_close = outputs.data.max(1)[1] # [bsz] cls_idx with max softmax value
            #logger.info(f"pred_close: {pred_close.shape}")
            unk_score = out_open[tmp_range, 0, pred_close] # [bsz]
            #logger.info(f"unk_score: {unk_score.shape}")
            known_score = outputs.max(1)[0] # [bsz] max softmax score
            #logger.info(f"known_score: {known_score.shape}")
            targets_unk = targets >= int(outputs.size(1)) # [bsz] True/False value (True for unk)
            #logger.info(f"targets_unk: {targets_unk.shape}")
            targets[targets_unk] = int(outputs.size(1)) # targets[targets_unk] = num_known_cls
            #logger.info(f"targets[targets_unk] = {int(outputs.size(1))}")
            known_targets = targets < int(outputs.size(1)) # [bsz] True/False value (True for known)
            #logger.info(f"known_targets: {known_targets.shape}")
            known_pred = outputs[known_targets] # [known_bsz, num_known_cls]
            #logger.info(f"known_pred: {known_pred.shape}")
            known_targets = targets[known_targets]
            #logger.info(f"known_targets: {known_targets.shape}")

            if len(known_pred) > 0:
                prec1, prec5 = accuracy(known_pred, known_targets, topk=(1, 5))
                top1.update(prec1.item(), known_pred.shape[0]) #所有known classes数据的top1 acc
                top5.update(prec5.item(), known_pred.shape[0]) #所有known classes数据的top5 acc

            ind_unk = unk_score > 0.5
            pred_close[ind_unk] = int(outputs.size(1))
            acc_all, unk_acc, size_unk = accuracy_open(pred_close,
                                                       targets,
                                                       num_classes=int(outputs.size(1))) #所有数据的acc, 成功拒绝outlier的acc, 当前batch中outlier数量
            acc.update(acc_all.item(), inputs.shape[0]) #所有数据（包括outlier应检测为outlier）的acc
            unk.update(unk_acc, size_unk) #outlier检测的acc

            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == 0:
                unk_all = unk_score
                known_all = known_score
                label_all = targets
            else:
                unk_all = torch.cat([unk_all, unk_score], 0)
                known_all = torch.cat([known_all, known_score], 0)
                label_all = torch.cat([label_all, targets], 0)

            test_loader.set_description("Test Iter: {batch:4}/{iter:4}. "
                                            "Data: {data:.3f}s."
                                            "Batch: {bt:.3f}s. "
                                            "Loss: {loss:.4f}. "
                                            "Closed t1: {top1:.3f} "
                                            "t5: {top5:.3f} "
                                            "acc: {acc:.3f}. "
                                            "unk: {unk:.3f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    acc=acc.avg,
                    unk=unk.avg,
                ))
        test_loader.close()
    ## ROC calculation
    #import pdb
    #pdb.set_trace()
    unk_all = unk_all.data.cpu().numpy()
    known_all = known_all.data.cpu().numpy()
    label_all = label_all.data.cpu().numpy()
    if not val:
        """roc = compute_roc(unk_all, label_all,
                          num_known=int(outputs.size(1)))
        roc_soft = compute_roc(-known_all, label_all,
                               num_known=int(outputs.size(1)))"""
        roc = 0
        roc_soft = 0
        ind_known = np.where(label_all < int(outputs.size(1)))[0]
        id_score = unk_all[ind_known]
        logger.info("Closed acc: {:.3f}".format(top1.avg)) #Closed acc: 所有known classes数据的top1 acc
        logger.info("Overall acc: {:.3f}".format(acc.avg)) #所有数据（包括outlier应检测为outlier）的acc
        logger.info("Unk acc: {:.3f}".format(unk.avg)) #outlier检测的acc
        logger.info("ROC: {:.3f}".format(roc))
        logger.info("ROC Softmax: {:.3f}".format(roc_soft))
        return losses.avg, top1.avg, acc.avg, \
               unk.avg, roc, roc_soft, id_score
    else:
        logger.info("Closed acc: {:.3f}".format(top1.avg))
        return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_open(pred, target, topk=(1,), num_classes=5):
    """Computes the precision@k for the specified values of k,
    num_classes are the number of known classes.
    This function returns overall accuracy,
    accuracy to reject unknown samples,
    the size of unknown samples in this batch."""
    maxk = max(topk)
    batch_size = target.size(0)
    pred = pred.view(-1, 1)
    pred = pred.t()
    ind = (target == num_classes)
    unknown_size = len(ind)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    if ind.sum() > 0:
        unk_corr = pred.eq(target).view(-1)[ind]
        acc = torch.sum(unk_corr).item() / unk_corr.size(0)
    else:
        acc = 0

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], acc, unknown_size


def compute_roc(unk_all, label_all, num_known):
    Y_test = np.zeros(unk_all.shape[0])
    unk_pos = np.where(label_all >= num_known)[0]
    Y_test[unk_pos] = 1
    return roc_auc_score(Y_test, unk_all)
