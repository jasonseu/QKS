# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2023-6-21
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2023 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
import time
import random
import traceback

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.utils import *
from lib.losses import *
from lib.metrics import *
from lib.dataset import MLDataset
from models.factory import create_model


class Trainer(object):
    def __init__(self, cfg, world_size, rank):
        super(Trainer, self).__init__()
        self.distributed = world_size > 1
        batch_size = cfg.batch_size // world_size if self.distributed else cfg.batch_size

        train_dataset = MLDataset(cfg.train_path, cfg, training=True)
        val_dataset = MLDataset(cfg.test_path, cfg, training=False)
        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            self.train_sampler = val_sampler = None
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(self.train_sampler is None),
                                       num_workers=4, sampler=self.train_sampler)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=val_sampler)

        torch.cuda.set_device(rank)
        self.model = create_model(cfg.model, cfg=cfg)
        self.model.cuda(rank)
        if self.distributed:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank], find_unused_parameters=True)

        parameters = self.model.parameters()
        self.loss_fn = get_loss_fn(cfg)
        self.optimizer = get_optimizer(parameters, cfg)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, cfg, steps_per_epoch=len(self.train_loader))

        self.cfg = cfg
        self.best_f1 = 0
        self.global_step = 0
        self.notdist_or_rank0 = (not self.distributed) or (self.distributed and rank == 0)
        if self.notdist_or_rank0:
            self.logger = get_logger(cfg.log_path, __name__)
            self.logger.info(train_dataset.transform)
            self.logger.info(val_dataset.transform)
            self.writer = SummaryWriter(log_dir=cfg.exp_dir)
            

    def run(self):
        patience = 0
        for epoch in range(self.cfg.max_epochs):
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
            self.train(epoch)
            f1 = self.validation(epoch)
            if self.cfg.lr_scheduler == 'ReduceLROnPlateau':
                self.lr_scheduler.step(f1)
            if self.best_f1 < f1 and self.notdist_or_rank0:
                torch.save(self.model.state_dict(), self.cfg.ckpt_best_path)
                self.best_f1 = f1
                patience = 0
            else:
                patience += 1
            if self.cfg.estop and patience > 2:
                break

        if self.notdist_or_rank0:
            self.logger.info('\ntraining over, best validation F1: {}'.format(self.best_f1))

    def train(self, epoch):
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)
        self.model.train()
        if self.cfg.bb_eval:
            self.model.clip_model.eval()
        for batch in self.train_loader:
            batch_begin = time.time()
            imgs = batch['img'].cuda()
            targets = batch['target'].cuda()
            with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                ret = self.model(imgs)
            
            logits = ret['logits']
            loss = self.loss_fn(logits, targets)
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            dur = time.time() - batch_begin
            if self.cfg.lr_scheduler == 'OneCycleLR':
                self.lr_scheduler.step()

            if self.global_step % (len(self.train_loader) // 10) == 0 and self.notdist_or_rank0:
                lr = get_lr(self.optimizer)
                self.writer.add_scalar('Loss/train', loss, self.global_step)
                self.writer.add_scalar('lr', lr, self.global_step)
                self.logger.info('TRAIN [epoch {}] loss: {:4f} lr:{:.6f} time:{:.4f}'.format(epoch, loss, lr, dur))

            self.global_step += 1

    @torch.no_grad()
    def validation(self, epoch):
        self.model.eval()
        scores, labels = [], []
        for batch in self.val_loader:
            images = batch['img'].cuda()
            _labels = batch['target'].cuda()
            logits = self.model(images)['logits']
            _scores = torch.sigmoid(logits).detach()
            if self.distributed:
                _scores = concat_all_gather(_scores)
                _labels = concat_all_gather(_labels)
            scores.append(_scores.cpu())
            labels.append(_labels.cpu())
        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)
        seen_scores = scores[:, :self.cfg.num_seen_labels]
        seen_labels = labels[:, :self.cfg.num_seen_labels]
        unseen_scores = scores[:, self.cfg.num_seen_labels:]
        unseen_labels = labels[:, self.cfg.num_seen_labels:]
        
        t1 = time.time()
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
        if self.notdist_or_rank0:
            self.writer.add_scalar('SEEN mAP/val', seen_mAP, self.global_step)
            self.logger.info("VALID [epoch {}] SEEN mAP: {:.4f}".format(epoch, seen_mAP))
        for k in self.cfg.topk:
            p, r, f1 = compute_F1(seen_scores, seen_labels, k_val=k)
            if self.notdist_or_rank0:
                self.logger.info("-- Top{} SEEN metrics P: {:.4f} R: {:.4f} F1: {:.4f}".format(k, p, r, f1))
        
        temp = unseen_labels != 0
        temp = torch.clamp(temp, 0, 1)
        mask = temp.sum(0).nonzero().flatten()
        unseen_labels = unseen_labels[:, mask]
        unseen_scores = unseen_scores[:, mask]

        score = unseen_mAP = compute_AP(unseen_scores, unseen_labels)
        if self.notdist_or_rank0:
            self.writer.add_scalar('ZSL mAP/val', unseen_mAP, self.global_step)
            self.logger.info("VALID [epoch {}] ZSL mAP: {:.4f}".format(epoch, unseen_mAP))
        for k in self.cfg.topk:
            p, r, f1 = compute_F1(unseen_scores, unseen_labels, k_val=k)
            score += f1
            if self.notdist_or_rank0:
                self.logger.info("-- Top{} ZSL metrics P: {:.4f} R: {:.4f} F1: {:.4f}".format(k, p, r, f1))
        
        temp = labels != 0
        temp = torch.clamp(temp, 0, 1)
        mask = temp.sum(0).nonzero().flatten()
        labels = labels[:, mask]
        scores = scores[:, mask]
        
        all_mAP = compute_AP(scores, labels)
        if self.notdist_or_rank0:
            self.writer.add_scalar('GZSL mAP/val', all_mAP, self.global_step)
            self.logger.info("VALID [epoch {}] GZSL mAP: {:.4f}".format(epoch, all_mAP))
        for k in self.cfg.topk:
            p, r, f1 = compute_F1(scores, labels, k_val=k)
            if self.notdist_or_rank0:
                self.logger.info("-- Top{} GZSL metrics P: {:.4f} R: {:.4f} F1: {:.4f}".format(k, p, r, f1))
                
        dur = (time.time() - t1) / 60
        print('Validation Time Cost: {:.4f}'.format(dur))
        
        return score


def main_worker(local_rank, ngpus_per_node, cfg, port=None):
    world_size = ngpus_per_node  # only single node is enough.
    if ngpus_per_node > 1:
        init_method = 'tcp://127.0.0.1:{}'.format(port)
        dist.init_process_group(backend='nccl', init_method=init_method, world_size=world_size, rank=local_rank)
    trainer = Trainer(cfg, world_size, local_rank)
    trainer.run()


if __name__ == "__main__":
    args = get_args()
    cfg = prepare_env(args, sys.argv)

    try:
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node > 1:
            port = 12345 + random.randint(0, 1000)
            setup_seed(cfg.seed)
            mp.spawn(main_worker, args=(ngpus_per_node, cfg, port,), nprocs=ngpus_per_node)
        else:
            setup_seed(cfg.seed)
            main_worker(0, ngpus_per_node, cfg)
    except (Exception, KeyboardInterrupt):
        print(traceback.format_exc())
        if not os.path.exists(cfg.ckpt_best_path):
            clear_exp(cfg.exp_dir)
