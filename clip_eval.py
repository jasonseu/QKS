import os
import argparse
from argparse import Namespace
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from lib.metrics import *
from lib.dataset import MLDataset
from models.clip import clip


template = 'There is a {} in the scene'


class Evaluator():
    def __init__(self, cfg):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load(cfg.arch, device=device)
        self.model.eval()
        dataset = MLDataset(cfg.test_path, cfg, training=False)
        self.data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
        self.labels = dataset.labels
        prompts = [template.format(label) for label in self.labels]
        texts = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            text_features = self.model.encode_text(texts)
        self.text_features = text_features / text_features.norm(dim=1, keepdim=True)
        self.cfg = cfg
        
    @torch.no_grad()
    def run(self):
        scores, labels = [], []
        gzsl_labels, zsl_labels = [], []
        for batch in tqdm(self.data_loader):
            imgs = batch['img'].cuda()
            targets = batch['target']
            img_path = batch['img_path']

            image_features = self.model.encode_image(imgs)[:, 0, :]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ self.text_features.t()
            # _scores = torch.sigmoid(logits).cpu()
            _scores = logits.cpu()
            scores.append(_scores)
            labels.append(targets)
            
            gzsl_topk_inds = np.argsort(-_scores.numpy())[:, :5]
            zsl_topk_inds = np.argsort(-_scores[:, self.cfg.num_seen_labels:].numpy())[:, :5]
            zsl_topk_inds = zsl_topk_inds + self.cfg.num_seen_labels
            for j in range(imgs.size(0)):
                img_name = os.path.basename(img_path[j])
                pred_labels = [self.labels[ind] for ind in gzsl_topk_inds[j]]
                gzsl_labels.append('{}\t{}\n'.format(img_name, '|'.join(pred_labels)))
                pred_labels = [self.labels[ind] for ind in zsl_topk_inds[j]]
                zsl_labels.append('{}\t{}\n'.format(img_name, '|'.join(pred_labels)))
                
        with open(os.path.join(self.cfg.exp_dir, 'prediction_clip_gzsl.txt'), 'w') as fw:
            fw.writelines(gzsl_labels)
        with open(os.path.join(self.cfg.exp_dir, 'prediction_clip_zsl.txt'), 'w') as fw:
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
    parser.add_argument('--data', type=str, default='nuswide')
    parser.add_argument('--arch', type=str, default='ViT-B/16', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16',
                                                                         'RN50x64', 'ViT-B/32', 'ViT-B/16',
                                                                         'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--mode', type=str, default='GZSL', choices=['ZSL', 'GZSL'])
    parser.add_argument('--topk', nargs='+', type=int, help='list of topk', default=[3, 5])
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--only-confidence-negative-label', '-only', action='store_true')

    args = parser.parse_args()
    cfg = vars(args)
    cfg['test_path'] = os.path.join('data', cfg['data'], 'test.txt')
    cfg['seen_label_path'] = os.path.join('data', cfg['data'], 'label_seen.txt')
    cfg['unseen_label_path'] = os.path.join('data', cfg['data'], 'label_unseen.txt')
    cfg['num_seen_labels'] = len([t.strip() for t in open(cfg['seen_label_path'])])
    cfg['topk'] = args.topk
    cfg['batch_size'] = args.batch_size
    cfg['exp_dir'] = 'experiments/vke2_nuswide/exp22'
    cfg = Namespace(**cfg)
    
    evaluator = Evaluator(cfg)
    evaluator.run()
