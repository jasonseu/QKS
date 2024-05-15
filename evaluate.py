# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2023-6-25
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2023 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import yaml
import argparse
from argparse import Namespace
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.factory import create_model
from lib.dataset import MLDataset
from lib.metrics import *


torch.backends.cudnn.benchmark = True


class Evaluator(object):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        dataset = MLDataset(cfg.test_path, cfg, training=False)
        self.dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
        self.labels = dataset.labels

        self.model = create_model(cfg.model, cfg=cfg)
        self.model.cuda()

        self.cfg = cfg

    @torch.no_grad()
    def run(self):
        model_dict = torch.load(self.cfg.ckpt_best_path)
        if list(model_dict.keys())[0].startswith('module'):
            model_dict = {k[7:]: v for k, v in model_dict.items()}
        self.model.load_state_dict(model_dict)
        print('loading best checkpoint success')

        self.model.eval()
        scores, labels = [], []
        gzsl_labels, zsl_labels = [], []
        zsl_token_cnt = torch.zeros(self.cfg.num_query_tokens)
        gzsl_token_cnt = torch.zeros(self.cfg.num_query_tokens)
        label_cnt = torch.zeros(self.cfg.num_seen_labels + self.cfg.num_unseen_labels)
        label_token_cnt = torch.zeros((self.cfg.num_seen_labels + self.cfg.num_unseen_labels, self.cfg.num_query_tokens))
        for batch in tqdm(self.dataloader):
            img = batch['img'].cuda()
            targets = batch['target']
            img_path = batch['img_path']
            ret = self.model(img)
            
            logits = ret['logits']
            _scores = torch.sigmoid(logits).detach().cpu()
            scores.append(_scores)
            labels.append(targets)
            
            label_cnt += (targets == 1).sum(0)
            
            indices = ret['indices'].cpu()  # batch_size x (num_seen_labels + num_unseen_labels)
            temp = F.one_hot(indices, num_classes=self.cfg.num_query_tokens)
            temp = (targets == 1).unsqueeze(-1) * temp  # only positive labels are taken into statistics
            zsl_token_cnt += temp.sum(1).sum(0)
            gzsl_token_cnt += temp[:, self.cfg.num_seen_labels:, :].sum(1).sum(0)
            label_token_cnt += temp.sum(0)
            
            gzsl_topk_inds = np.argsort(-_scores.numpy())[:, :5]
            zsl_topk_inds = np.argsort(-_scores[:, self.cfg.num_seen_labels:].numpy())[:, :5]
            zsl_topk_inds = zsl_topk_inds + self.cfg.num_seen_labels
            for j in range(img.size(0)):
                img_name = os.path.basename(img_path[j])
                pred_labels = [self.labels[ind] for ind in gzsl_topk_inds[j]]
                gzsl_labels.append('{}\t{}\n'.format(img_name, '|'.join(pred_labels)))
                pred_labels = [self.labels[ind] for ind in zsl_topk_inds[j]]
                zsl_labels.append('{}\t{}\n'.format(img_name, '|'.join(pred_labels)))
                
        np.save(os.path.join(self.cfg.exp_dir, 'label_cnt.npy'), label_cnt.numpy())
        np.save(os.path.join(self.cfg.exp_dir, 'token_zsl_cnt.npy'), zsl_token_cnt.numpy())
        np.save(os.path.join(self.cfg.exp_dir, 'token_gzsl_cnt.npy'), gzsl_token_cnt.numpy())
        np.save(os.path.join(self.cfg.exp_dir, 'label_token_cnt.npy'), label_token_cnt.numpy())
                
        with open(os.path.join(self.cfg.exp_dir, 'prediction_gzsl.txt'), 'w') as fw:
            fw.writelines(gzsl_labels)
        with open(os.path.join(self.cfg.exp_dir, 'prediction_zsl.txt'), 'w') as fw:
            fw.writelines(zsl_labels)
                
        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)
        seen_scores = scores[:, :self.cfg.num_seen_labels]
        seen_labels = labels[:, :self.cfg.num_seen_labels]
        unseen_scores = scores[:, self.cfg.num_seen_labels:]
        unseen_labels = labels[:, self.cfg.num_seen_labels:]
        
        mask = torch.clamp(seen_labels, 0, 1).sum(1).nonzero().flatten()  # take only the images with positive annotations
        seen_scores = seen_scores[mask]
        seen_labels = seen_labels[mask]
        mask = torch.clamp(unseen_labels, 0, 1).sum(1).nonzero().flatten()  # take only the images with positive annotations
        unseen_scores = unseen_scores[mask]
        unseen_labels = unseen_labels[mask]
        mask = torch.clamp(labels, 0, 1).sum(1).nonzero().flatten()  # take only the images with positive annotations
        scores = scores[mask]
        labels = labels[mask]
        
        temp = seen_labels != 0
        temp = torch.clamp(temp, 0, 1)
        mask = temp.sum(0).nonzero().flatten()
        seen_labels = seen_labels[:, mask]
        seen_scores = seen_scores[:, mask]
        
        seen_mAP = compute_AP(seen_scores, seen_labels)
        print("SEEN mAP: {:.4f}".format(seen_mAP))
        for k in self.cfg.topk:
            p, r, f1 = compute_F1(seen_scores, seen_labels, k_val=k)
            print("-- Top{} SEEN metrics P: {:.4f} R: {:.4f} F1: {:.4f}".format(k, p, r, f1))
        
        temp = unseen_labels != 0
        temp = torch.clamp(temp, 0, 1)
        mask = temp.sum(0).nonzero().flatten()
        unseen_labels = unseen_labels[:, mask]
        unseen_scores = unseen_scores[:, mask]

        unseen_mAP = compute_AP(unseen_scores, unseen_labels)
        print("ZSL mAP: {:.4f}".format(unseen_mAP))
        for k in self.cfg.topk:
            p, r, f1 = compute_F1(unseen_scores, unseen_labels, k_val=k)
            print("-- Top{} ZSL metrics P: {:.4f} R: {:.4f} F1: {:.4f}".format(k, p, r, f1))
        
        temp = labels != 0
        temp = torch.clamp(temp, 0, 1)
        mask = temp.sum(0).nonzero().flatten()
        labels = labels[:, mask]
        scores = scores[:, mask]
        
        all_mAP = compute_AP(scores, labels)
        print("GZSL mAP: {:.4f}".format(all_mAP))
        for k in self.cfg.topk:
            p, r, f1 = compute_F1(scores, labels, k_val=k)
            print("-- Top{} GZSL metrics P: {:.4f} R: {:.4f} F1: {:.4f}".format(k, p, r, f1))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, default='experiments/vke2_nuswide/exp22')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--topk', nargs='+', type=int, default=[3, 5])
    args = parser.parse_args()
    cfg_path = os.path.join(args.exp_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError('config file not found in the {}!'.format(cfg_path))
    cfg = yaml.load(open(cfg_path, 'r'))
    cfg = Namespace(**cfg)
    cfg.batch_size = args.batch_size
    cfg.topk = args.topk
    print(cfg)

    evaluator = Evaluator(cfg)
    evaluator.run()
