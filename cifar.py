import logging
import math
import os
import datetime
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from cifar_dataset import CIFAR100_txt, Mydataset_test, TransformOpenMatch, x_u_split, imagenet_split, TransformFixMatch \
                        , MyDataset_labeled, MyDataset_unlabeled

logger = logging.getLogger(__name__)

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

"""
# used args paramters
args.base_task_cls: base task cls number [check]
args.steps: incremental steps [check]
args.save_root: root path to save datasets [check]
args.root: root path of original datasets [check]
"""

def create_classIL_task(args):

    ##############################################################################
    #error check
    logger.info(f" Error Check before creating datasets")
    if args.base_task_cls != 0:
        if (100 - args.base_task_cls) % args.steps != 0:
            raise ValueError('cifar100 classes cannot be split into tasks')
    else:
        if 100 % args.steps != 0:
            raise ValueError('cifar100 classes cannot be split into tasks')

    ##############################################################################
    # set class order
    logger.info(f" Setting class order")
    task_list = [i for i in range(100)]
    np.random.shuffle(task_list)
    task_list = [68, 56, 78, 8, 23, 84, 90, 65, 74, 76,
                 40, 89, 3, 92, 55, 9, 26, 80, 43, 38,
                 58, 70, 77, 1, 85, 19, 17, 50, 28, 53,
                 13, 81, 45, 82, 6, 59, 83, 16, 15, 44,
                 91, 41, 72, 60, 79, 52, 20, 10, 31, 54,
                 37, 95, 14, 71, 96, 98, 97, 2, 64, 66,
                 42, 22, 35, 86, 24, 34, 87, 21, 99, 0,
                 88, 27, 18, 94, 11, 12, 47, 25, 30, 46,
                 62, 69, 36, 61, 7, 63, 75, 5, 32, 4,
                 51, 48, 73, 93, 39, 67, 29, 49, 57, 33, 100]
    logger.info(f" task list: {task_list}")

    # write class order into file
    
    save_dir = os.path.join(args.save_root, 'cifar100_B%i_%isteps_%s' % (args.base_task_cls, args.steps, args.timestamp))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    task_checkfile = open(save_dir + '/checkfile.txt', mode='w')
    if args.base_task_cls != 0:
        task_checkfile.write('task 1: ')
        for k in range(args.base_task_cls):
            task_checkfile.write(str(task_list[k]) + ' ')
        task_checkfile.write('\n')
        class_per_task = int((100 - args.base_task_cls) // args.steps)
        for t in range(args.steps):
            task_checkfile.write('task %i: '%(t+2))
            for k in range(class_per_task):
                task_checkfile.write(str(task_list[args.base_task_cls+t*class_per_task+k]) + ' ')
            task_checkfile.write('\n')
    else:
        class_per_task = int(100 // args.steps)
        for t in range(args.steps):
            task_checkfile.write('task %i: '%(t+1))
            for k in range(class_per_task):
                task_checkfile.write(str(task_list[t*class_per_task+k]) + ' ')
            task_checkfile.write('\n')

    ##############################################################################
    # create labeled & unlabeled train dataset txt for each step

    logger.info(f" Creating files for each steps")
    cifar100_txt = CIFAR100_txt(args.root)
    logger.info(f" Finishing loading data")

    if args.base_task_cls != 0:
        # if having a large base task
        
        # task 1
        logger.info(f" Starting Task 1")
        current_class = task_list[:args.base_task_cls]
        future_class = task_list[args.base_task_cls:]
        labeled_idx, unlabeled_idx = x_u_split(args, cifar100_txt.train_label, current_class, future_class)
        l_train = open(save_dir + '/train_1_labeled.txt', mode='w')
        u_train = open(save_dir + '/train_1_unlabeled.txt', mode='w')
        l_train_data, l_train_label = cifar100_txt.train_data[labeled_idx], cifar100_txt.train_label[labeled_idx]
        u_train_data, u_train_label = cifar100_txt.train_data[unlabeled_idx], cifar100_txt.train_label[unlabeled_idx]
        
        # write labeled datafile
        for i in range(len(l_train_data)):
            l_train.write(l_train_data[i] + ' ' + str(l_train_label[i]) + '\n')
        # write unlabeled datafile
        for i in range(len(u_train_data)):
            u_train.write(u_train_data[i] + ' ' + str(u_train_label[i]) + '\n')

        #task 2~n
        cls_per_task = int((100 - args.base_task_cls) // args.steps)
        for t in range(args.steps):
            logger.info(f" Starting Task {t+2}")
            old_class = task_list[:args.base_task_cls+t*cls_per_task]
            current_class = task_list[args.base_task_cls+t*cls_per_task:args.base_task_cls+(t+1)*cls_per_task]
            future_class = task_list[args.base_task_cls+(t+1)*cls_per_task:]
            labeled_idx, unlabeled_idx = x_u_split(args, cifar100_txt.train_label, current_class, future_class, old_class)
            l_train = open(save_dir + '/train_%i_labeled.txt' % (t+2), mode='w')
            u_train = open(save_dir + '/train_%i_unlabeled.txt' % (t+2), mode='w')
            l_train_data, l_train_label = cifar100_txt.train_data[labeled_idx], cifar100_txt.train_label[labeled_idx]
            u_train_data, u_train_label = cifar100_txt.train_data[unlabeled_idx], cifar100_txt.train_label[unlabeled_idx]
        
            # write labeled datafile
            for i in range(len(l_train_data)):
                l_train.write(l_train_data[i] + ' ' + str(l_train_label[i]) + '\n')
            # write unlabeled datafile
            for i in range(len(u_train_data)):
                u_train.write(u_train_data[i] + ' ' + str(u_train_label[i]) + '\n')

    else:
        #if not having a large base task

        #task1~n
        cls_per_task = int(100 // args.steps)
        for t in range(args.steps):
            logger.info(f" Starting Task {t+1}")
            old_class = task_list[:t*cls_per_task]
            current_class = task_list[t*cls_per_task:(t+1)*cls_per_task]
            future_class = task_list[(t+1)*cls_per_task:]
            labeled_idx, unlabeled_idx = x_u_split(args, cifar100_txt.train_label, current_class, future_class, old_class)
            l_train = open(save_dir + '/train_%i_labeled.txt' % (t+1), mode='w')
            u_train = open(save_dir + '/train_%i_unlabeled.txt' % (t+1), mode='w')
            l_train_data, l_train_label = cifar100_txt.train_data[labeled_idx], cifar100_txt.train_label[labeled_idx]
            u_train_data, u_train_label = cifar100_txt.train_data[unlabeled_idx], cifar100_txt.train_label[unlabeled_idx]
        
            # write labeled datafile
            for i in range(len(l_train_data)):
                l_train.write(l_train_data[i] + ' ' + str(l_train_label[i]) + '\n')
            # write unlabeled datafile
            for i in range(len(u_train_data)):
                u_train.write(u_train_data[i] + ' ' + str(u_train_label[i]) + '\n')

    ##############################################################################
    # create test dataset txt for the whole task

    #1. CIFAR100 test data
    logger.info(f" Creating CIFAR100 Test file")
    test = open(save_dir + '/test_cifar100.txt', mode='w')
    for i in range(len(cifar100_txt.test_data)):
                test.write(cifar100_txt.test_data[i] + ' ' + str(cifar100_txt.test_label[i]) + '\n')

    # 2. imagenet valid data
    logger.info(f" Creating ImageNet Valid file")
    val_idx = imagenet_split(args, cifar100_txt.imagenet_val_label)
    #val_data, val_label = cifar100_txt.imagenet_val_data[val_idx], cifar100_txt.imagenet_val_label[val_idx]
    val_data = cifar100_txt.imagenet_val_data[val_idx]
    valid = open(save_dir + '/valid_imagenet.txt', mode='w')
    for i in range(len(val_data)):
        valid.write(val_data[i] + ' ' + str(100) + '\n')

def create_train_task_openmatch(args):

    """
    return: 2 datasets for 1 step, augmented in openmatch form
    """
    #transform_fix = TransformFixMatch(mean=cifar100_mean, std=cifar100_std) #return: (weak_aug, strong_aug, weaker_aug)
    transform_open = TransformOpenMatch(mean=cifar100_mean, std=cifar100_std) #return: (weak_aug1, weak_aug2, weaker_aug)
    #use OpenMatch Augment
    l_train_dataset = MyDataset_labeled(args, transforms=transform_open)
    u_train_dataset = MyDataset_unlabeled(args, transforms=transform_open)
    if args.local_rank == 0:
        logger.info(f"l_train_dataset length: {l_train_dataset.__len__()}")
        logger.info(f"u_train_dataset length: {u_train_dataset.__len__()}")
    # !!!!!: when training, the subset copied from u_train_dataset must change its augment operation into TransformOpenMatch


    return l_train_dataset, u_train_dataset

def create_test_dataset(args):

    """
    return: test dataset: cifar100 test data
            valid dataset: imagenet sampled data for ood detection
    """

    transform = transforms.Compose([
            transforms.Scale((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
        ]) #return: no_aug

    test_dataset, valid_dataset = \
    Mydataset_test(args, transforms=transform, test_not_val=True), Mydataset_test(args, transforms=transform, test_not_val=False)
    if args.local_rank == 0:
        logger.info(f"test_dataset length: {test_dataset.__len__()}")
        logger.info(f"valid_dataset length: {valid_dataset.__len__()}")

    return test_dataset, valid_dataset
