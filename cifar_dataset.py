import os
import numpy as np
import math
from torchvision import transforms
from torch.utils.data import Dataset
from randaugment import RandAugmentMC
from PIL import Image
import logging
import cv2

"""
# used args paramters
args.num_labeled [check]
args.expand_labels [check]
args.num_expand_x [check]
args.memory_size [check]
args.unlabeled_mu [check]
args.imagenet_valid_size [check]
args.timestamp: time which created the using dataset [check]
args.save_root [check]

"""
logger = logging.getLogger(__name__)

imagenet_cls_num = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                    450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 
                    461, 462, 463, 470, 471, 472, 473, 474, 475, 476, 477, 478, 
                    800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 
                    811, 812, 813, 814, 815, 816] # 50 classes in total, imagenet id start from 0

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

class CIFAR100_txt():
    def __init__(self, root):

        ##############################################################################
        # load all cifar100 filepath in txt
        dataset_dir = os.path.join(root,'CIFAR100')
        
        # 1. train set
        self.train_data = []
        self.train_label = []
        for filename in os.listdir(os.path.join(dataset_dir,'train')):
            if filename.endswith('.png'):
                label = int(filename.split('.')[0].split('-')[2])
                self.train_data.append(str(os.path.join(os.path.join(dataset_dir,'train'), filename)))
                self.train_label.append(label)
        
        # 2. test set
        self.test_data = []
        self.test_label = []
        for filename in os.listdir(os.path.join(dataset_dir,'test')):
            if filename.endswith('.png'):
                label = int(filename.split('.')[0].split('-')[2])
                self.test_data.append(str(os.path.join(os.path.join(dataset_dir,'test'), filename)))
                self.test_label.append(label)
        
        ##############################################################################
        # load all imagenet filepath in txt
        # cls ids which are used for the task is defined at the very top of this file ()
        
        #TODO: fill in needed cls ids
        
        # 1. train set
        #self.imagenet_train_data = []
        #self.imagenet_train_label = []
        f = open(os.path.join(root, 'imagenet1k', 'train.txt'),'r')
        lines = f.read().splitlines()
        for line in lines:
            filepath, label = line.split(' ')
            if int(label) in imagenet_cls_num:
                #self.imagenet_train_data.append(os.path.join(root, 'imagenet1k', 'train', filepath))
                #self.imagenet_train_label.append(100)
                self.train_data.append(os.path.join(root, 'imagenet1k', 'train', filepath))
                self.train_label.append(100)
        
        # 2. valid set
        self.imagenet_val_data = []
        self.imagenet_val_label = []
        f = open(os.path.join(root, 'imagenet1k', 'val.txt'),'r')
        lines = f.read().splitlines()
        for line in lines:
            filepath, label = line.split(' ')
            if int(label) in imagenet_cls_num:
                self.imagenet_val_data.append(os.path.join(root, 'imagenet1k', 'val', filepath))
                self.imagenet_val_label.append(int(label))
        
        ##############################################################################
        # convert to ndarray
        self.train_data = np.array(self.train_data)
        self.train_label = np.array(self.train_label)
        self.test_data = np.array(self.test_data)
        self.test_label = np.array(self.test_label)
        self.imagenet_val_data = np.array(self.imagenet_val_data)
        self.imagenet_val_label = np.array(self.imagenet_val_label)

def x_u_split(args, labels, current_cls, future_cls, old_cls=[]):

    """
    split & sample all train_data into labeled & unlabeled data for CIFAR100 TASK
    """

    ##############################################################################
    # setup
    label_per_class = args.num_labeled #// args.num_classes
    labels = np.array(labels)
    if old_cls != []:
        mem_per_cls = args.memory_size // len(old_cls) # sample per old cls
        fut_num = (args.unlabeled_mu * args.num_labeled * len(current_cls) - args.memory_size) # total size for unknown cls
    else:
        fut_num = args.unlabeled_mu * args.num_labeled * len(current_cls)
    labeled_idx = []
    unlabeled_idx = []

    ##############################################################################
    # split current cls
    
    for i in current_cls:
        idx = np.where(labels == i)[0]
        l_idx = np.random.choice(idx, label_per_class, False)
        l_indices = np.argwhere(np.isin(idx, l_idx))
        u_idx = np.delete(idx, l_indices)
        labeled_idx.extend(l_idx)
        unlabeled_idx.extend(u_idx)

    ##############################################################################
    # sample old cls
    for i in old_cls:
        idx = np.where(labels == i)[0]
        u_idx = np.random.choice(idx, mem_per_cls, False)
        unlabeled_idx.extend(u_idx)
    
    ##############################################################################
    # sample future cls
    all_future_idx = []
    for i in future_cls:
        idx = np.where(labels == i)[0]
        all_future_idx.extend(idx)
    u_idx = np.random.choice(all_future_idx, fut_num, False)
    unlabeled_idx.extend(u_idx)

    ##############################################################################
    # expand labeled dataset for efficiency
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)

    if args.expand_labels:
        num_expand_x = args.num_expand_x #expand dataset to reduce times of dataloading
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
        unlabeled_idx = np.hstack([unlabeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    np.random.shuffle(unlabeled_idx)

    return labeled_idx, unlabeled_idx

    ##############################################################################
    # TODO: small scale test on this function to make sure classes are separated rightly

def imagenet_split(args, labels):

    """
    sample imagenet val data for CIFAR100 TASK
    """

    ##############################################################################
    # setup
    img_per_class = int(args.imagenet_valid_size // len(imagenet_cls_num)) #// args.num_classes
    labels = np.array(labels)
    valid_idx = []

    ##############################################################################
    # sample used cls
    for i in imagenet_cls_num:
        idx = np.where(labels == i)[0]
        l_idx = np.random.choice(idx, img_per_class, False)
        valid_idx.extend(l_idx)

    return valid_idx

    ##############################################################################
    # TODO: small scale test on this function to make sure classes are separated rightly

class MyDataset_labeled(Dataset):
    def __init__(self, args, transforms=None):
        super(MyDataset_labeled,self).__init__()

        ##############################################################################
        # setup
        f = open(os.path.join(args.save_root, 'cifar100_B%i_%isteps_%s' % (args.base_task_cls, args.steps, args.timestamp), 
                'train_%i_labeled.txt' % (args.now_step)), 'r')
        self.data = []
        self.label = []
        self.transform = transforms

        ##############################################################################
        # load data
        data = f.read().splitlines()
        for line in data:
            if line.startswith('/'):
                sep = line.split()
                """img = Image.open(sep[0])
                if img.mode != 'RGB':
                    img = img.convert('RGB')"""
                img = cv2.imread(sep[0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.data.append(img)
                self.label.append(int(task_list.index(int(sep[1])))) # origin label --> shuffled label
        
        f.close()
    
    def __getitem__(self, index):

        img, label = self.data[index], self.label[index]
        #img = self.transform(img)
        img = self.transform(Image.fromarray(img))

        return img, label
    
    def __len__(self):

        return len(self.label)

class MyDataset_unlabeled(Dataset):
    def __init__(self, args, transforms=None):
        super(MyDataset_unlabeled,self).__init__()
        ##############################################################################
        # setup
        f = open(os.path.join(args.save_root, 'cifar100_B%i_%isteps_%s' % (args.base_task_cls, args.steps, args.timestamp), 
                'train_%i_unlabeled.txt' % (args.now_step)), 'r')
        self.data_all = []
        self.label_all = []
        self.data_select = []
        self.label_select = []
        self.transform = transforms
        self.idx = []

        ##############################################################################
        # load data
        data = f.read().splitlines()
        for line in data:
            if line.startswith('/'):
                sep = line.split()
                """img = Image.open(sep[0])
                if img.mode != 'RGB':
                    img = img.convert('RGB')"""
                img = cv2.imread(sep[0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.data_all.append(img)
                self.label_all.append(int(task_list.index(int(sep[1])))) # origin label --> shuffled label
        #self.init_index()
        f.close()
        self.data_all = np.array(self.data_all)
        self.label_all = np.array(self.label_all)
        self.init_index()
    
    def init_index(self):

        self.data_select, self.label_select = self.data_all, self.label_all
    
    def select_idx(self, idx):

        self.data_select, self.label_select = self.data_all[idx], self.label_all[idx]
        self.idx = idx

    def __getitem__(self, index):

        img, label = self.data_select[index], self.label_select[index]
        #img = self.transform(img)
        img = self.transform(Image.fromarray(img))

        return img, label
    
    def __len__(self):

        return len(self.label_select)

class Mydataset_test(Dataset):
    def __init__(self, args, transforms=None, test_not_val=True):
        super(Mydataset_test,self).__init__()
        ##############################################################################
        # setup
        
        self.data = []
        self.label = []
        self.transform = transforms
        self.test_not_val = test_not_val

        ##############################################################################
        # load data
        if self.test_not_val:
            # 1. cifar100 test
            f_test = open(os.path.join(args.save_root, 'cifar100_B%i_%isteps_%s' % (args.base_task_cls, args.steps, args.timestamp), 
                    'test_cifar100.txt'), 'r')
            data = f_test.read().splitlines()
            for line in data:
                if line.startswith('/'):
                #TODO: check if really start with '/'
                    sep = line.split()
                    """img = Image.open(sep[0])
                    if img.mode != 'RGB':
                        img = img.convert('RGB')"""
                    img = cv2.imread(sep[0])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.data.append(img)
                    self.label.append(int(task_list.index(int(sep[1])))) # origin label --> shuffled label
            f_test.close()
        else:
            # 2. imagenet valid
            f_val = open(os.path.join(args.save_root, 'cifar100_B%i_%isteps_%s' % (args.base_task_cls, args.steps, args.timestamp), 
                'valid_imagenet.txt'), 'r')
            data = f_val.read().splitlines()
            for line in data:
                if line.startswith('/'):
                #TODO: check if really start with '/'
                    sep = line.split()
                    """img = Image.open(sep[0])
                    if img.mode != 'RGB':
                        img = img.convert('RGB')"""
                    img = cv2.imread(sep[0])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.data.append(img)
                    self.label.append(int(task_list.index(int(sep[1])))) # origin label --> shuffled label
        
            f_val.close()

    def __getitem__(self, index):

        img, label = self.data[index], self.label[index]
        #img = self.transform(img)
        img = self.transform(Image.fromarray(img))

        return img, label
    
    def __len__(self):

        return len(self.label)
    
class TransformFixMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.Scale((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.Scale((32,32)),
            transforms.RandomHorizontalFlip(),])
        self.strong = transforms.Compose([
            transforms.Scale((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.weak(x))
        else:
            return weak, strong

class TransformOpenMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.Scale((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.Scale((32,32)),
            transforms.RandomHorizontalFlip(),])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.weak(x)

        if self.norm:
            return self.normalize(weak), self.normalize(self.weak(x)), self.normalize(strong)
        else:
            return weak, strong