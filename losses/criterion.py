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
import copy
import torch.nn as nn
from pytorch_metric_learning import losses
from tqdm import tqdm

class ProxyConcat_Loss(nn.Module):
    def __init__(self, K_g, K_p):
        super(ProxyConcat_Loss, self).__init__()
        self.K_g            = K_g
        self.K_p            = K_p
        self.SupCon_loss    = losses.SupConLoss(temperature=0.05).to(torch.device('cuda'))
        
    def forward(self, probes, gallery, probes_target, gallery_target):
        gallery       = gallery.reshape(gallery.shape[0], gallery.shape[2]*gallery.shape[1])
        probes        = probes.reshape(probes.shape[0], probes.shape[2]*probes.shape[1])
        print(gallery.shape, probes.shape)
        
        embeddings    = torch.cat([probes, gallery], dim = 0)
        labels        = torch.cat([probes_target, gallery_target], dim = 1).squeeze()
        supcon_loss   = self.SupCon_loss(embeddings, labels)
        return supcon_loss

def generate_lattice_points(dim=3, N=4):
    if (N - 1 > dim):
        print("Error: Dimensionality lesser than number of classes + 1")
        return 
    
    print("Computing Lattice Points")
    points = [[0,0],[1,0],[0.5,np.sqrt(3)/2]]

    for j in tqdm(range(2, N-1)):
        temppoints = copy.deepcopy(points)
        centroid = points[-1][-1] / len(points)
        temppoints[-1][-1] = centroid
        new_last_coordinate_last_axis = np.sqrt(1 - sum([i**2 for i in temppoints[-1]]))
        for point in points:
            point.append(0)
        temp = temppoints[-1] + [new_last_coordinate_last_axis]
        points.append(temp)
            
    for k in range(dim - N + 1):
        for point in points:
            point.append(0)
            
    hyper_pyramid_center = np.mean(points, axis=0)
    points = np.asarray(points) - hyper_pyramid_center
    points = [point / np.linalg.norm(point) for point in points]

    vector_norms = []
    distances = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            a = np.asarray(points[i])
            b = np.asarray(points[j])
            dist = np.linalg.norm(a-b)
            distances.append(dist)
        vector_norms.append(np.linalg.norm(a))
    
    print("Verification of distances: ", all(x-distances[0] <= 0.0000000001 for x in distances), distances[0])    
    print("Verification of norms: ", all(x-vector_norms[0] <= 0.0000000001 for x in vector_norms), vector_norms[0] )    
    with open('lattice_' + "_".join([str(i) for i in np.asarray(points).shape]) + '.npy', 'wb') as f:
        print("Lattice Saved to:", 'lattice_' + "_".join([str(i) for i in np.asarray(points).shape]) + '.npy')
        np.save(f, np.asarray(points))

    return np.asarray(points)   

class LatticeLoss(nn.Module):
    def __init__(self, N, D):
        super(LatticeLoss, self).__init__()
        print("Lattice Loss")
        self.thresh = 0.1
        self.scale_neg = 40.0
        self.scale_pos = 5.0
        lattice_points = generate_lattice_points(D, N)
        print("Generated Lattice Points")
        self.lattice_points = torch.from_numpy(lattice_points).float().cuda()
        self.lattice_points.requires_grad = False
            
    def forward(self, proxies):
        lattice_points    = F.normalize(self.lattice_points, dim=-1)
        proxies           = F.normalize(proxies, dim=-1)         
        similarities      = F.linear(lattice_points, proxies)     
        
        pos_mask    = torch.eye(similarities.shape[0]).cuda() # identity matrix
        neg_mask    = 1 - pos_mask
        pos_exp     = torch.exp(-self.scale_pos*(similarities-self.thresh))
        neg_exp     = torch.exp( self.scale_neg*(similarities-self.thresh))
        P_exp       = torch.where(pos_mask == 1,pos_exp,torch.zeros_like(pos_exp)).sum(dim=-1)
        N_exp       = torch.where(neg_mask == 1,neg_exp,torch.zeros_like(neg_exp)).sum(dim=-1)
        pos_loss    = torch.log(1+P_exp).sum()/self.scale_pos
        neg_loss    = torch.log(1+N_exp).sum()/self.scale_neg
        losspg      = pos_loss + neg_loss
        return losspg