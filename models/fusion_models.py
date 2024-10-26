from itertools import combinations
import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import  torch.nn.functional as F
import numpy as np
import torch
from PIL import Image
import cv2
import pandas as pd
from PIL import Image
from torchvision import transforms as trans
import pickle
import torch.nn as nn


class ProxyFusion(nn.Module):
    def __init__(self, DIM=512):
        super(ProxyFusion, self).__init__()
        self.softmax_temp_gl = 1.0
        self.softmax_temp_pb = 1.0  
        
        self.K_g_all                 = 11 # 8, 4
        self.K_p_all                 = 11
        self.domain_dim              = 10
        self.K_g                     = 4
        self.K_p                     = 4
        
        self.pooler = nn.ModuleList([nn.Sequential(nn.Linear(DIM * 3, DIM * 2), 
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=0.7),
                                    nn.Linear(DIM * 2, DIM * 2), 
                                    nn.LeakyReLU(), 
                                    nn.Dropout(p=0.7),
                                    nn.Linear(DIM * 2, DIM),
                                    nn.Dropout(p=0.7),
                                    nn.LeakyReLU()
                                    ) for i in range(self.K_g_all)])
                
        self.pooler.apply(self.init_weights)
        self.pooler.requires_grad = True
        
        self.transform           = nn.Linear(DIM, self.domain_dim)
        nn.init.kaiming_normal_(self.transform.weight, mode='fan_out')
        self.transform.weight.requires_grad_()
        
        self.proxies_g           = nn.Parameter(torch.full((self.K_g_all, DIM), 1e-5))
        nn.init.kaiming_normal_(self.proxies_g, mode='fan_out')
        self.proxies_g.requires_grad = True
        
        self.proxies_p           = nn.Parameter(torch.full((self.K_p_all, DIM), 1e-5))
        nn.init.kaiming_normal_(self.proxies_p, mode='fan_out')
        self.proxies_p.requires_grad = True
        
        self.probe_linear = nn.Linear(DIM,DIM)
        nn.init.eye_(self.probe_linear.weight)
        nn.init.zeros_(self.probe_linear.bias)
        self.probe_linear.weight.requires_grad_()

    def attention_probes(self, feat, proxies): # feat: Feature Set of the video
        normed_feat    = F.normalize(feat, dim=-1) 
        # feat = N x D
        # proxies = K x D
        K = proxies.shape[0]
        transformed_feat    = self.transform(feat)
        transformer_proixes = self.transform(proxies)
        transformed_feat    = F.normalize(transformed_feat, dim=-1)             # Normalize features - N x d'
        transformer_proixes = F.normalize(transformer_proixes, dim=-1)          # Normalize proxies  - K x d'

        # Calculate similarities (N x K)
        similarities        = F.linear(transformed_feat, transformer_proixes)   # N x K                
        global_similarities = similarities.sum(dim=0)                           # 1 x K dimensional                        
        _, top_indices      = torch.topk(global_similarities, self.K_p)         # Select indices of top_kp (Top 4)
        
        context_vectors     = []                                                # context vectors -> prototypes 
        for i in top_indices:
            summary_vector  = torch.cat([feat.mean(dim=0), torch.var(feat, dim=0, unbiased=False), proxies[i]], dim=-1) # [mean + var + proxy]
            context_vectors.append(self.pooler[i](summary_vector))                             # Each 1 x D, 4 - 1 x D
        
        context_vector      = torch.vstack(context_vectors)                                    # Context Vector 4 x D
        attention_scores    = F.linear(self.probe_linear(feat), context_vector)                # MatMul - 4 x N similarities, I = Featureset N x D, W = 4 x D => 4 x N (I, W)
        weights             = F.softmax(self.softmax_temp_pb * attention_scores, dim=0)        # 4 x N weights
        agg_feat            = torch.sum(normed_feat.unsqueeze(2) * weights.unsqueeze(1),dim=0)        # (4 x N) X (N x D) => 4 x D aggreagated feature vector
        return torch.permute(agg_feat, (1,0)), similarities
        
    def attention_gallery(self, feat, proxies):
        if(len(feat.shape) == 3):
            feat = feat.reshape(feat.shape[0] * feat.shape[1], feat.shape[2])
        normed_feat    = F.normalize(feat, dim=-1) 
        # print(feat.shape, proxies.shape)
        K = proxies.shape[0]
        transformed_feat    = self.transform(feat)
        transformer_proixes = self.transform(proxies)
        transformed_feat    = F.normalize(transformed_feat, dim=-1)         # Normalize features
        transformer_proixes = F.normalize(transformer_proixes, dim=-1)      # Normalize proxies

        # Calculate similarities (N x K)
        similarities        = F.linear(transformed_feat, transformer_proixes)
        global_similarities = similarities.sum(dim=0)                       # K dimensional
        _, top_indices      = torch.topk(global_similarities, self.K_p)

        context_vectors     = []
        for i in top_indices:
            summary_vector      = torch.cat([feat.mean(dim=0), torch.var(feat, dim=0, unbiased=False), proxies[i]], dim=-1)
            context_vectors.append(self.pooler[i](summary_vector))
        
        context_vector      = torch.vstack(context_vectors)
        attention_scores    = F.linear(feat, context_vector)                # N x sK    
        weights             = F.softmax(self.softmax_temp_pb * attention_scores, dim=0) 
        agg_feat            = torch.sum(normed_feat.unsqueeze(2) * weights.unsqueeze(1),dim=0)
        return torch.permute(agg_feat, (1,0)), similarities # torch.cat([torch.permute(agg_feat, (1,0)), attended_emb.unsqueeze(dim=0)], dim=0), 

    def eval_fuse_probe(self, feat):
        return self.attention_probes(feat, self.proxies_p)[0]

    def eval_fuse_gallery(self, feat):
        return self.attention_gallery(feat, self.proxies_g)[0]

    def init_weights(self, layer):
        stddev=0.001
        if type(layer) == nn.Linear:
            noise = torch.randn_like(layer.weight) * stddev
            layer.weight.data.zero_()
            layer.weight.data.add_(noise)
            layer.bias.data.zero_()

    def forward(self, probes, gallery, probe_lengths, gallery_lengths):
        final_probe_features    = []
        final_gallery_features  = []
        probe_proxy_sims        = []
        gallery_proxy_sims      = []
        
        probes                  = probes.squeeze(0)
        gallery                 = gallery.squeeze(0)
        probe_lengths           = probe_lengths.squeeze(0)
        gallery_lengths         = gallery_lengths.squeeze(0)

        for i in range(probes.shape[0]):
            lp                  = probe_lengths[i]
            probe_feat          = probes[i,0:lp,:]
            agg_probe, p_sim    = self.attention_probes(probe_feat, self.proxies_p)
            final_probe_features.append(agg_probe)
            probe_proxy_sims.append(p_sim)

        for i in range(gallery.shape[0]):
            lg                  = gallery_lengths[i]
            gallery_feat        = gallery[i,0:lg,:]
            agg_gallery, g_sim  = self.attention_gallery(gallery_feat, self.proxies_g)
            final_gallery_features.append(agg_gallery)
            gallery_proxy_sims.append(g_sim)
            
        final_probe_features    = torch.stack(final_probe_features,dim=0)
        final_gallery_features  = torch.stack(final_gallery_features,dim=0)

        return final_probe_features, self.transform(self.proxies_p), final_gallery_features, self.transform(self.proxies_g)
