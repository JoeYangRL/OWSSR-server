import sys
sys.path.append("..")
import logging
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from cifar import TransformOpenMatch, cifar100_std, cifar100_mean, normal_mean, \
    create_train_task_openmatch, create_test_dataset
from tqdm import tqdm
from cifar_dataset import TransformFixMatch
from utils import AverageMeter,\
    save_checkpoint, \
    reduce_tensor, select_memory, split_u_dataset, select_confirm

from loss import ova_loss, ova_ent, dist_loss
from test import test


"""
# used args paramters
args.online_distill: set to True when updating memory set every epoch [check]
args.eval_step: iters train for one epoch
args.mu_u [check]
args.mu_c [check]
args.mu_m [check]
args.lambda_oem [check]
args.lambda_socr [check]
args.lambda_dist [check]
args.start_fix [check]
"""

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0

def train(args, model, old_model, ema_model, optimizer, scheduler, rank):
    
    global best_acc
    global best_acc_val

    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_o = AverageMeter()
    losses_oem = AverageMeter()
    losses_socr = AverageMeter()
    losses_fix = AverageMeter()
    losses_dist = AverageMeter()
    mask_probs = AverageMeter()
    end = time.time()

    if rank == 0:
        logger.info("---------- set datasets -------------------")
    if args.world_size > 1:
        l_epoch = 0
        u_epoch = 0
        c_epoch = 0
        m_epoch = 0
    
    l_train_dataset, u_train_dataset = create_train_task_openmatch(args)
    test_dataset, valid_dataset = create_test_dataset(args)
    confirm_dataset = copy.deepcopy(u_train_dataset) # for fixmatch
    confirm_dataset.transform = TransformFixMatch(mean=cifar100_mean, std=cifar100_std)
    if args.now_step > 1:
        memory_dataset = copy.deepcopy(u_train_dataset) # for memory replay
    
    
    if (rank == 0) and (args.now_step > 1):
        select_memory(args, old_model, u_train_dataset, memory_dataset)
    
    train_sampler = RandomSampler if args.world_size == 1 else DistributedSampler
    
    #TODO: check each dataset's transform function
    
    """test_trainloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False)
    valid_trainloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False)"""
        
    #test_iter = test_trainloader.__iter__()    
    #valid_iter = valid_trainloader.__iter__()

    if rank == 0:
        logger.info("---------- start defining metrics -------------------")
    default_out = "Epoch: {now_epoch}/{epoch:4}. " \
                  "LR: {now_lr:.6f}. " \
                  "Lab: {loss_x:.4f}. " \
                  "Open: {loss_o:.4f}"
    output_args = vars(args)
    default_out += " OEM  {loss_oem:.4f}"
    default_out += " SOCR  {loss_socr:.4f}"
    default_out += " Fix  {loss_fix:.4f}"
    default_out += " Dist  {loss_dist:.4f}"

    

    if rank == 0:
        logger.info("----------start training -------------------")

    for epoch in range(0, args.epoch):
        model.train()
        output_args["now_epoch"] = epoch
        if rank == 0:
            p_bar = tqdm(range(args.eval_step), disable=False)
        
        if (args.warmup_epoch > 0) and (epoch < args.warmup_epoch):
            lr = args.lr * (epoch+1) / args.warmup_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        ##############################################################################
        # set dataset & dataloader
        if rank == 0:
            logger.info("---------- start defining dataloaders -------------------")

        if (epoch >= args.start_fix) and (rank == 0):
            if args.now_step > 1:
                if args.online_distill:
                    # reselect memory & confirmed data per epoch
                    if args.use_ema:
                        split_u_dataset(args, ema_model.ema, u_train_dataset, memory_dataset, confirm_dataset)
                    else:
                        split_u_dataset(args, model, u_train_dataset, memory_dataset, confirm_dataset)
                else:
                    # select confirmed data and keep memory data excluded
                    if args.use_ema:
                        select_confirm(args, ema_model.ema, u_train_dataset, confirm_dataset, memory_dataset)
                    else:
                        select_confirm(args, model, u_train_dataset, confirm_dataset, memory_dataset)
            else:
                # select confirmed unlabeled data for step 1
                if args.use_ema:
                    select_confirm(args, ema_model.ema, u_train_dataset, confirm_dataset)
                else:
                    select_confirm(args, model, u_train_dataset, confirm_dataset)
        
        u_bsz = args.batch_size * args.mu_u
        c_bsz = args.batch_size * args.mu_c
        
        l_train_trainloader = DataLoader(
            l_train_dataset,
            sampler=train_sampler(l_train_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True)
        u_train_trainloader = DataLoader(
            u_train_dataset,
            sampler=train_sampler(u_train_dataset),
            batch_size=u_bsz,
            num_workers=args.num_workers,
            drop_last=True)
        confirm_trainloader = DataLoader(
            confirm_dataset,
            sampler=train_sampler(confirm_dataset),
            batch_size=c_bsz,
            num_workers=args.num_workers,
            drop_last=True)
        l_train_iter = l_train_trainloader.__iter__()
        u_train_iter = u_train_trainloader.__iter__()
        confirm_iter = confirm_trainloader.__iter__()
        if args.now_step > 1:
            m_bsz = args.batch_size * args.mu_m
            memory_trainloader = DataLoader(
                memory_dataset,
                sampler=train_sampler(memory_dataset),
                batch_size=m_bsz,
                num_workers=args.num_workers,
                drop_last=True)
            memory_iter = memory_trainloader.__iter__()
    

        for batch_idx in range(args.eval_step):
            
            ##############################################################################
            # Data loading

            # 1. labeled data
            # TransformOpenMatch
            try:
                (_, l_x_weak, l_x), l_y = l_train_iter.__next__()
            except:
                if args.world_size > 1:
                    l_epoch += 1
                    l_train_trainloader.sampler.set_epoch(l_epoch)
                l_train_iter = l_train_trainloader.__iter__()
                (_, l_x_weak, l_x), l_y = l_train_iter.__next__()
            
            # 2. rest unlabeled data
            # TransformOpenMatch
            try:
                (u_x_weak1, u_x_weak2, _), _ = u_train_iter.__next__()
            except:
                if args.world_size > 1:
                    u_epoch += 1
                    u_train_trainloader.sampler.set_epoch(u_epoch)
                u_train_iter = u_train_trainloader.__iter__()
                (u_x_weak1, u_x_weak2, _), _ = u_train_iter.__next__()
            
            # 3. confirm data
            # TransformFixMatch
            if epoch >= args.start_fix:

                try:
                    (_, c_x_weak, c_x_strong), _ = confirm_iter.__next__()
                except:
                    if args.world_size > 1:
                        c_epoch += 1
                        confirm_trainloader.sampler.set_epoch(c_epoch)
                    confirm_iter = confirm_trainloader.__iter__()
                    (_, c_x_weak, c_x_strong), _ = confirm_iter.__next__()
            
            # 4. memory data
            # TransformOpenMatch
            if args.now_step > 1:
                try:
                    (_, m_x_weak, _), _ = memory_iter.__next__()
                except:
                    if args.world_size > 1:
                        m_epoch += 1
                        memory_trainloader.sampler.set_epoch(m_epoch)
                    memory_iter = memory_trainloader.__iter__()
                    (_, m_x_weak, _), _ = memory_iter.__next__()
            
            # concat all the data (ORDER: labeled --> rest_unlabeled --> confirmed --> memory)
            if epoch >= args.start_fix:
                x_all = torch.cat([l_x, l_x_weak, u_x_weak1, u_x_weak2, c_x_weak, c_x_strong], 0)
            else:
                x_all = torch.cat([l_x, l_x_weak, u_x_weak1, u_x_weak2], 0)
            if args.now_step > 1:
                x_all = torch.cat([x_all, m_x_weak], 0) # l_bsz + 2*u_bsz + 2*c_bsz + m_bsz

            data_time.update(time.time() - end)

            x_all = x_all.to(args.device)
            l_y = l_y.to(args.device)

            ##############################################################################
            # Forward & computing loss
            logits, logits_open = model(x_all) # logits: [bsz, num_cls]; logits_open: [bsz, 2*num_cls]
            if epoch >= args.start_fix:
                logits_open_u1, logits_open_u2 = logits_open[2*args.batch_size:2*args.batch_size+2*u_bsz].chunk(2)
            else:
                logits_open_u1, logits_open_u2 = logits_open[2*args.batch_size:2*args.batch_size+2*u_bsz].chunk(2)

            # Loss for labeled samples
            Lx = F.cross_entropy(logits[:2*args.batch_size], l_y.repeat(2),reduction='mean')
            Lo = ova_loss(logits_open[:2*args.batch_size], l_y.repeat(2))

            # Open-set entropy minimization
            L_oem = ova_ent(logits_open_u1) / 2.
            L_oem += ova_ent(logits_open_u2) / 2.

            # Soft consistenty regularization
            logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
            logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
            logits_open_u1 = F.softmax(logits_open_u1, 1)
            logits_open_u2 = F.softmax(logits_open_u2, 1)
            L_socr = torch.mean(torch.sum(torch.sum(torch.abs(
                logits_open_u1 - logits_open_u2)**2, 1), 1))

            # FixMatch loss
            if epoch >= args.start_fix:
                logits_u_w, logits_u_s = logits[2*(args.batch_size+u_bsz):2*(args.batch_size+u_bsz+c_bsz)].chunk(2)
                pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1).to(args.device)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).to(args.device).float()
                L_fix = (F.cross_entropy(logits_u_s,
                                         targets_u,
                                         reduction='none') * mask).to(args.device).mean()
                mask_probs.update(mask.mean().item())
            else:
                L_fix = torch.zeros(1).to(args.device).mean()
            
            # diatillation loss
            if args.now_step > 1:
                old_x = torch.cat([l_x_weak, u_x_weak1], 0)
                logits_old, logits_open_old = old_model(old_x.to(args.device))
                logits_m, logits_open_m = logits[args.batch_size:(2*args.batch_size+u_bsz),:-args.cls_per_step], logits_open[args.batch_size:(2*args.batch_size+u_bsz),:-2*args.cls_per_step]
                L_dist = dist_loss(args, logits_m, logits_open_m, logits_old, logits_open_old)
            else:
                L_dist = torch.zeros(1).to(args.device).mean()

            loss = Lx + Lo + args.lambda_oem * L_oem + args.lambda_socr * L_socr + L_fix + args.lambda_dist * L_dist
            loss.backward()

            #同步多卡上的结果
            if args.world_size > 1:
                reduce_loss = reduce_tensor(loss.data)
                reduce_Lx = reduce_tensor(Lx.data)
                reduce_Lo = reduce_tensor(Lo.data)
                reduce_L_oem = reduce_tensor(L_oem.data)
                reduce_L_socr = reduce_tensor(L_socr.data)
                reduce_L_fix = reduce_tensor(L_fix.data)
                reduce_L_dist = reduce_tensor(L_dist.data)
            else:
                reduce_loss = loss.data
                reduce_Lx = Lx.data
                reduce_Lo = Lo.data
                reduce_L_oem = L_oem.data
                reduce_L_socr = L_socr.data
                reduce_L_fix = L_fix.data
                reduce_L_dist = L_dist.data
            
            # avgmeter更新数据
            losses.update(reduce_loss.item())
            losses_x.update(reduce_Lx.item())
            losses_o.update(reduce_Lo.item())
            losses_oem.update(reduce_L_oem.item())
            losses_socr.update(reduce_L_socr.item())
            losses_fix.update(reduce_L_fix.item())
            losses_dist.update(reduce_L_dist.item())

            #将最新的平均值传入output
            output_args["batch"] = batch_idx
            output_args["loss_x"] = losses_x.avg
            output_args["loss_o"] = losses_o.avg
            output_args["loss_oem"] = losses_oem.avg
            output_args["loss_socr"] = losses_socr.avg
            output_args["loss_fix"] = losses_fix.avg
            output_args["loss_dist"] = losses_dist.avg
            output_args["now_lr"] = [group["lr"] for group in optimizer.param_groups][0]

            optimizer.step()
            torch.distributed.barrier()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()

            if rank == 0:
                p_bar.set_description(default_out.format(**output_args))
                p_bar.update()

        if rank == 0:
            logger.info(p_bar)
            p_bar.close()
        

        if rank == 0:

            valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False)
            test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False)
            #val_acc = test(args, valid_loader, model, val=True)
            if args.use_ema:
                test_model = ema_model.ema
            else:
                test_model = model
            test_loss, test_acc_close, test_overall, \
            test_unk, test_roc, test_roc_softm, test_id \
                = test(args, test_loader, test_model)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_o', losses_o.avg, epoch)
            args.writer.add_scalar('train/4.train_loss_oem', losses_oem.avg, epoch)
            args.writer.add_scalar('train/5.train_loss_socr', losses_socr.avg, epoch)
            args.writer.add_scalar('train/6.train_loss_fix', losses_fix.avg, epoch)
            args.writer.add_scalar('train/7.train_loss_dist', losses_dist.avg, epoch)
            args.writer.add_scalar('train/8.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc_close, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc_close > best_acc_val
            best_acc_val = max(test_acc_close, best_acc_val)
            if is_best:
                overall_valid = test_overall
                close_valid = test_acc_close
                unk_valid = test_unk
                roc_valid = test_roc
                roc_softm_valid = test_roc_softm
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc close': test_acc_close,
                'acc overall': test_overall,
                'unk': test_unk,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.output_path, args.now_step)
            test_accs.append(test_acc_close)
            logger.info('Best val closed acc: {:.3f}'.format(best_acc_val)) #验证集中最佳closed acc
            logger.info('Valid closed acc: {:.3f}'.format(close_valid)) #测试集中最佳closed acc
            logger.info('Valid overall acc: {:.3f}'.format(overall_valid)) #测试集中最佳overall acc (包括unk数据)
            logger.info('Valid unk acc: {:.3f}'.format(unk_valid)) #测试集中最佳unk acc
            logger.info('Valid roc: {:.3f}'.format(roc_valid)) #测试集中最佳roc
            logger.info('Valid roc soft: {:.3f}'.format(roc_softm_valid)) #测试集中最佳roc soft
            logger.info('Mean top-1 acc: {:.3f}\n'.format(
                np.mean(test_accs[-20:])))
    
        if epoch >= args.warmup_epoch:
            scheduler.step()

    if rank == 0:
        args.writer.close()
