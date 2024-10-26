import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
import torch
from PIL import Image
import cv2
import pandas as pd
from PIL import Image
from torchvision import transforms as trans
import pickle
from random import shuffle
from tqdm import tqdm
import h5py

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

class BRSDataset(Dataset):
    """
    Load precomputed probe and gallery features
    """
    def __init__(self, args, data_path, subjects):
        self.subjects = subjects
        self.data_path = data_path
        self.subid_to_id_mapping = {k: i for i,k in enumerate(subjects)}
        self.args = args
        
    def __len__(self):
        return 1

    def __getitem__(self, index):
        lengths_probe        = []
        lengths_gallery      = []
        probes_list          = []
        gallery_list         = [] 
        gallery_target       = []
        probes_target        = []

        subjects             = random.sample(self.subjects, self.args.subjects_per_batch)
        subjects             = [item for item in subjects for i in range(self.args.subject_repeat_factor)] 
        random.shuffle(subjects)
        
        for i,each in enumerate(tqdm(subjects)):
            with h5py.File(self.data_path + "/features_" + str(each) + ".hdf5", 'r') as file:
                data                    = file[each]
                num_probes              = len(data.keys()) - 2
                selected_probe          = 'probe_' + str(random.randint(0,num_probes))
                try:
                    feat_probe              = data[selected_probe][:]
                except:
                    print(data.keys(), selected_probe)
                lengths_probe.append(feat_probe.shape[0])
                probes_list.append(feat_probe)
                probes_target.append(self.subid_to_id_mapping[each])

                selected_gallery_media  = []
                gallery_media           = data['gallery']
                
                num_gallery_videos      = len(gallery_media['video'].keys())
                selected_gallery_videos = str(random.randint(0,num_gallery_videos - 1))
                selected_gallery_media.extend([['video', selected_gallery_videos]])


                num_gallery_images      = len(gallery_media['images'].keys())
                selected_gallery_images = [['images', str(i)] for i in random.sample(range(0, num_gallery_images-1), k=random.randint(0, 4))]
                selected_gallery_media.extend(selected_gallery_images)

                gallery_media_feat      = []
                for i, each_media in enumerate(selected_gallery_media):
                    try:
                        feat_gallery = data['gallery'][each_media[0]][each_media[1]][:]
                    except:
                        print(data['gallery'][each_media[0]].keys(), each_media[1])
                    gallery_media_feat.append(feat_gallery)
                
                gallery_media_feat = np.concatenate(gallery_media_feat,axis=0)
                gallery_list.append(gallery_media_feat)
                lengths_gallery.append(gallery_media_feat.shape[0])
                gallery_target.append(self.subid_to_id_mapping[each])

        max_length_probe = max(lengths_probe)
        max_length_gallery = max(lengths_gallery)

        padded_probes = []
        padded_gallerys = []

        for each_probe in probes_list:
            probe_tensor = torch.from_numpy(each_probe)
            pad_vector = torch.zeros((max_length_probe-probe_tensor.shape[0],self.args.feature_dim))
            probe_tensor = torch.cat((probe_tensor,pad_vector),0)
            padded_probes.append(probe_tensor)
        
        for each_gallery in gallery_list:
            gallery_tensor = torch.from_numpy(each_gallery)
            pad_vector = torch.zeros((max_length_gallery-gallery_tensor.shape[0],self.args.feature_dim))
            gallery_tensor = torch.cat((gallery_tensor,pad_vector),0)
            padded_gallerys.append(gallery_tensor)

        probes = torch.stack(padded_probes,dim=0)
        gallery = torch.stack(padded_gallerys,dim=0)
        pad_index_probes = torch.Tensor(lengths_probe).long()
        pad_index_gallery = torch.Tensor(lengths_gallery).long()

        probes_target = torch.Tensor(probes_target).long()
        gallery_target = torch.Tensor(gallery_target).long()

        return probes,gallery,pad_index_probes,pad_index_gallery, probes_target, gallery_target