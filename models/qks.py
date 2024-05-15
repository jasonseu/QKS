# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2023-6-21
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2023 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import torch
import torch.nn as nn
import numpy as np

from .clip import clip
from .utils import BertConfig, BertModel
from .factory import register_model, create_backbone


__all__ = ['qks']


class QKS(nn.Module):
    def __init__(self, clip_model, vision_width, cfg):
        super().__init__()
        self.cfg = cfg
        self.vision_width = vision_width
        self.clip_model = clip_model
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.proj = self.clip_model.visual.proj
        self.clip_model.visual.proj = None
        self.proj.requires_grad = True
        prompt_templates = [t.strip() for t in open(self.cfg.prompt_template_path)]
        
        if os.path.exists(cfg.train_text_feat_path):  # load features from cache
            self.train_text_features = torch.from_numpy(np.load(cfg.train_text_feat_path)).cuda()
        else:
            seen_labels = [label.strip() for label in open(cfg.seen_label_path)]
            self.train_text_features = self.prompt_engineering(seen_labels, prompt_templates)
            np.save(cfg.train_text_feat_path, self.train_text_features.cpu().numpy())
        
        if os.path.exists(cfg.test_text_feat_path):  # load features from cache
            self.test_text_features = torch.from_numpy(np.load(cfg.test_text_feat_path)).cuda()
        else:
            unseen_labels = [label.strip() for label in open(cfg.unseen_label_path)]
            unseen_text_features = self.prompt_engineering(unseen_labels, prompt_templates)
            self.test_text_features = torch.concat([self.train_text_features, unseen_text_features], dim=0)
            np.save(cfg.test_text_feat_path, self.test_text_features.cpu().numpy())
        
        self.net, self.query_tokens = self.init_QKSNet()
        
    
    @torch.no_grad()
    def prompt_engineering(self, labels, prompt_templates):
        zeroshot_weights = []
        for label in labels:
            texts = [template.format(label) for template in prompt_templates]
            texts = clip.tokenize(texts).cuda()
            text_features = self.clip_model.encode_text(texts)
            text_features = text_features.mean(0)
            zeroshot_weights.append(text_features)
        
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
        return zeroshot_weights
    
    def init_QKSNet(self, cross_attention_freq=1):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = self.vision_width
        # insert cross-attention layer every other block
        encoder_config.output_attentions = True
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.num_hidden_layers = self.cfg.num_hidden_layers
        encoder_config.query_length = self.cfg.num_query_tokens
        vke_net = BertModel(encoder_config, add_pooling_layer=False)
        query_tokens = nn.Parameter(torch.zeros(1, self.cfg.num_query_tokens, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return vke_net, query_tokens
    
    @property
    def text_features(self):
        if self.training:
            return self.train_text_features
        else:
            return self.test_text_features
    
    def forward(self, x):
        image_features = self.clip_model.encode_image(x)
        if self.cfg.rmcls:
            image_features = image_features[:, 1:, :]
        image_atts = torch.ones(image_features.size()[:-1], dtype=torch.long).to(image_features.device)
        
        query_tokens = self.query_tokens.expand(image_features.shape[0], -1, -1)
        query_output = self.net(
            query_embeds=query_tokens,
            encoder_hidden_states=image_features,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )
        query_features = query_output.last_hidden_state @ self.proj
        logits = query_features @ self.text_features.t()
        cos_similarity = logits.transpose(1, 2).softmax(dim=-1)
        logits, indices = logits.max(dim=1)
        # logits, indices = torch.topk(logits, self.cfg.K, dim=1)
        # logits = torch.sum(logits, dim=1)
        
        return {
            'logits': logits,
            'indices': indices,
            'cos_similarity': cos_similarity,
            'att_weights': query_output.cross_attentions
        }
        

@register_model
def qks(cfg):
    clip_model, feat_dim = create_backbone(cfg.arch)
    model = QKS(clip_model, feat_dim, cfg)
    return model
