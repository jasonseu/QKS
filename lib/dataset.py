# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2023-6-21
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2023 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from randaugment import RandAugment
from .cutout import CutoutPIL, SLCutoutPIL
from models.clip.clip import _transform
import pandas as pd

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


logger = logging.getLogger(__name__)


class MLDataset(Dataset):
    def __init__(self, data_path, cfg, training=True):
        super(MLDataset, self).__init__()
        if training:
            self.labels = [line.strip('\n') for line in open(cfg.seen_label_path)]
        else:
            unseen_labels = [line.strip('\n') for line in open(cfg.unseen_label_path)]
            if cfg.mode == 'ZSL':
                self.labels = unseen_labels
            elif cfg.mode == "GZSL":
                seen_labels = [line.strip('\n') for line in open(cfg.seen_label_path)]
                self.labels = seen_labels + unseen_labels
        
        self.num_classes = len(self.labels)
        self.label2id = {label: i for i, label in enumerate(self.labels)}

        self.data = pd.read_csv(data_path).astype(object)
        # if training:
        #     self.data.dropna(axis=0, subset=['posi_labels'], inplace=True)
        if not training:
            if cfg.mode == 'none':
                self.data.rename(columns={'seen_posi_labels': 'posi_labels'}, inplace=True)
                self.data.rename(columns={'seen_nega_labels': 'nega_labels'}, inplace=True)
                self.data.drop(['unseen_posi_labels', 'unseen_nega_labels'], axis=1, inplace=True)
            elif cfg.mode == 'ZSL':
                self.data.rename(columns={'unseen_posi_labels': 'posi_labels'}, inplace=True)
                self.data.rename(columns={'unseen_nega_labels': 'nega_labels'}, inplace=True)
                self.data.drop(['seen_posi_labels', 'seen_nega_labels'], axis=1, inplace=True)
            elif cfg.mode == 'GZSL':
                posi_columns = ['seen_posi_labels', 'unseen_posi_labels']
                nega_columns = ['seen_nega_labels', 'unseen_nega_labels']
                self.data['posi_labels'] = self.data[posi_columns].apply(lambda x: x.str.cat(sep='|'), axis=1)
                self.data['nega_labels'] = self.data[nega_columns].apply(lambda x: x.str.cat(sep='|'), axis=1)
                self.data.drop([*posi_columns, *nega_columns], axis=1, inplace=True)
                self.data.replace('', np.nan, inplace=True)
            
        self.data.dropna(axis=0, subset=['posi_labels'], inplace=True)  # take only the images with positive labels
        
        self.cfg = cfg
        self.training = training
        self.transform = self.get_transform()

    def get_transform(self):
        if self.training:
            mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
            return transforms.Compose([
                transforms.Resize(self.cfg.img_size, interpolation=BICUBIC),
                transforms.CenterCrop(self.cfg.img_size),
                CutoutPIL(cutout_factor=0.5),
                transforms.RandomHorizontalFlip(),
                RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        return _transform(self.cfg.img_size)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img_path = row['image_path']
        posi_labels = row['posi_labels']
        nega_labels = row['nega_labels']
        
        target = np.zeros(self.num_classes)
        posi_labels = [] if pd.isna(posi_labels) else [self.label2id[t] for t in posi_labels.split('|')]
        target[posi_labels] = 1
        if self.cfg.data == 'nuswide':
            target = target * 2 - 1
        else:
            nega_labels = [] if pd.isna(nega_labels) else [self.label2id[t] for t in nega_labels.split('|')]
            target[nega_labels] = -1
        
        img_data = Image.open(img_path).convert('RGB')
        img_data = self.transform(img_data)

        item = {
            'img': img_data,
            'target': target,
            'img_path': img_path,
        }
        
        return item
        
        
    def __len__(self):
        return len(self.data)
