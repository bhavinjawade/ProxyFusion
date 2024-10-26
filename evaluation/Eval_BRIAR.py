import os
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
from lxml import etree
import xmltodict
from tqdm import tqdm
import glob
BUFFER_MODIFIER = 0.2
LARGE_FIGURE = (16,16)
import sys
outer_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, outer_directory)
from utils.VerificationResults import VerificationResults
import matplotlib
import matplotlib.pyplot as plt
from models.fusion_models import ProxyFusion
import csv
import scipy
import pickle as pkl
import argparse

def metrics_at_thersholds(dataframe):
    tar = dataframe['tar']
    far = dataframe['far']
    tar = list(dataframe['tar'])
    far = list(dataframe['far'])
    perfStat = [ 0, 0, 0, 0, 0]
    for i in range(len(far)):
        if i > 0:
            if far[i] > 1e-5 and far[i-1] < 1e-5:
                perfStat[4] = (tar[i]+tar[i-1])/2
            elif far[i] > 1e-4 and far[i-1] < 1e-4:
                perfStat[3] = (tar[i]+tar[i-1])/2
            elif far[i] > 1e-3 and far[i-1] < 1e-3:
                perfStat[2] = (tar[i]+tar[i-1])/2
            elif far[i] > 1e-2 and far[i-1] < 1e-2:
                perfStat[1] = (tar[i]+tar[i-1])/2
            elif far[i] > 1e-1 and far[i-1] < 1e-1:
                perfStat[0] = (tar[i]+tar[i-1])/2
    return perfStat

def normalizeRows(x):    
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)    
    x = x / (x_norm + 1e-8)
    return x
  
def fuse_function(media, model, type = "probe", featureset = None, axis=0, keepdims=True, baseline = False):
    
    if (np.isnan(np.sum(featureset))):
        featureset = np.zeros((1, 4, 512)) * np.nan
        return featureset
    
    if baseline == True:
        feat = np.mean(featureset,axis=0,keepdims=True)
    else:
        if type == "gallery":
            feat = model.module.eval_fuse_gallery(torch.from_numpy(featureset).cuda()).detach().cpu().numpy()[np.newaxis, ...]
        elif type == "probe":
            feat = model.module.eval_fuse_probe(torch.from_numpy(featureset).cuda()).detach().cpu().numpy()[np.newaxis, ...]

    return feat

def Evaluate_Model(model, args, baseline=True, ax=None, linestyle='dashed', color='red', probe_aggregate_type = None, gallery_aggregate_type = None):
    
    print("Loading protocol information")
    
    eval_probe              = args.bts_protocol_path + '/analysis_main/Probe_BTS_briar-rd_ALL.csv' # Change csv filename here to select face_incl_trt, face_incl_ctrl, face_restr_trt, face_restr_ctrl
    eval_probe_df           = pd.read_csv(eval_probe)
    eval_probe_df           = eval_probe_df[eval_probe_df["face_incl_trt"] == True]
    media_paths             = eval_probe_df['media_path']
    entry_ids               = list(eval_probe_df['entry_id'])


    g2_csv                  = args.bts_protocol_path + '/analysis_main/Gallery2.csv' # Change csv filename here to select face_incl_trt, face_incl_ctrl, face_restr_trt, face_restr_ctrl
    g2_csv                  = pd.read_csv(g2_csv)
    g2_ids                  = set(g2_csv['subjectId'])

    g1_csv                  = args.bts_protocol_path + '/analysis_main/Gallery1.csv' # Change csv filename here to select face_incl_trt, face_incl_ctrl, face_restr_trt, face_restr_ctrl
    g1_csv                  = pd.read_csv(g1_csv)
    g1_ids                  = set(g1_csv['subjectId'])

    print("Load all file paths")

    method = args.face_feature_extractor

    GALLERY_1_DIR_FACE      = args.bts_embeddings_path + '/' + args.face_detector + '/' + args.face_feature_extractor + '/gallery1/'
    GALLERY_2_DIR_FACE      = args.bts_embeddings_path + '/' + args.face_detector + '/' + args.face_feature_extractor +  '/gallery2/'
    PROBE_DIR_FACE          = args.bts_embeddings_path + '/' + args.face_detector + '/' + args.face_feature_extractor +  '/probe/'

    with open("<Path to distractors identity names for duplication across gallery 1 and gallery 2>/bts3_distractors.txt") as file:
        distractors = [f.strip() for f in file.readlines()]

    gallery_1_files_face    = glob.glob(GALLERY_1_DIR_FACE + '*.pickle')
    gallery_2_files_face    = glob.glob(GALLERY_2_DIR_FACE + '*.pickle') + distractors
    probe_files_face        = glob.glob(PROBE_DIR_FACE + '*.pickle')

    print(len(gallery_1_files_face), len(gallery_2_files_face), len(glob.glob(GALLERY_2_DIR_FACE + '*.pickle')), len(probe_files_face))

    probe_media     = {} # mapping from media name to file paths
    gallery_1_media = {} # mapping from gallery identity to media name to file paths
    gallery_2_media = {} # mapping from gallery identity to media name to file paths

    for i, media_path in enumerate(media_paths):
        media_name    = media_path.split("/")[-1].split(".")[0]
        entry_id      = entry_ids[i]
        template_name = entry_id + "-" + media_name
        if (PROBE_DIR_FACE + template_name + '.pickle') in probe_files_face:
            probe_media[template_name] = [PROBE_DIR_FACE + template_name + '.pickle']
        else:
            probe_media[template_name] = [None]

    for filename in gallery_1_files_face:
        media_name = filename.split("/")[-1].split(".pickle")[0]
        gid = media_name.split("_")[1].strip()
        if (gid in g1_ids):
            if gid in gallery_1_media:
                if media_name in gallery_1_media[gid]:
                    gallery_1_media[gid][media_name].append(filename)
                else:
                    gallery_1_media[gid][media_name] = [filename]
            else:
                gallery_1_media[gid] = {}
                gallery_1_media[gid][media_name] = [filename]
            
    for filename in gallery_2_files_face:
        media_name = filename.split("/")[-1].split(".pickle")[0]
        gid = media_name.split("_")[1].strip()
        if (gid in g2_ids):
            if gid in gallery_2_media:
                if media_name in gallery_2_media[gid]:
                    gallery_2_media[gid][media_name].append(filename)
                else:
                    gallery_2_media[gid][media_name] = [filename]
            else:
                gallery_2_media[gid] = {}
                gallery_2_media[gid][media_name] = [filename]
            
    print("Gallery 1 and Gallery 2 identities and Probe medias:", len(gallery_1_media.keys()), len(gallery_2_media.keys()), len(probe_media.keys()))
    print("Number of None Probes: ", len([v[0] for k,v in probe_media.items() if v[0] == None]))
    print("Loading embeddings from pickle files")

    missing_probes = 0
    probe_embeddings_nonaggregated = {}
    gall1_embeddings_nonaggregated = {}
    gall2_embeddings_nonaggregated = {}

    print("Gallery 2 =============================")

    for id in tqdm(gallery_2_media.keys()):
        for media, filepaths in gallery_2_media[id].items():
            for filepath in filepaths:
                with open(filepath,'rb') as f:
                    gallery_data_frame = pickle.load(f)
                    try:         
                        gallery_embedding  = np.stack(gallery_data_frame['feature'],axis=0)
                    except:
                        print(filepath)
                    # gallery_embedding  = gallery_embedding / np.linalg.norm(gallery_embedding, axis=1)[:, np.newaxis]
                if id in gall2_embeddings_nonaggregated:
                    if media in gall2_embeddings_nonaggregated[id]:
                        gall2_embeddings_nonaggregated[id][media].append(gallery_embedding)
                    else:
                        gall2_embeddings_nonaggregated[id][media] = [gallery_embedding]
                else:
                    gall2_embeddings_nonaggregated[id] = {media:[gallery_embedding]}
    with open("./retina_face_gall2_embeddings_nonaggregated.pickle", "wb") as file:
        pickle.dump(gall2_embeddings_nonaggregated, file)
    
    print("Gallery 1 =============================")

    for id in tqdm(gallery_1_media.keys()):
        for media, filepaths in gallery_1_media[id].items():
            for filepath in filepaths:
                with open(filepath,'rb') as f:
                    gallery_data_frame = pickle.load(f) 
                    gallery_embedding  = np.stack(gallery_data_frame['feature'],axis=0)
                    # gallery_embedding  = gallery_embedding / np.linalg.norm(gallery_embedding, axis=1)[:, np.newaxis]
                if id in gall1_embeddings_nonaggregated:
                    if media in gall1_embeddings_nonaggregated[id]:
                        gall1_embeddings_nonaggregated[id][media].append(gallery_embedding)
                    else:
                        gall1_embeddings_nonaggregated[id][media] = [gallery_embedding]
                else:
                    gall1_embeddings_nonaggregated[id] = {media:[gallery_embedding]}
    with open("./retina_face_gall1_embeddings_nonaggregated.pickle", "wb") as file:
        pickle.dump(gall1_embeddings_nonaggregated, file)

    print("Probes =============================")

    for media, files in tqdm(probe_media.items()):
        if (files == [None]):
            feat = np.zeros((1,512)) * np.nan
            missing_probes += 1
            probe_embeddings_nonaggregated[media] = [feat]
        else:
            for filepath in files:
                with open(filepath,'rb') as f:
                    probe_data_frame = pickle.load(f)                
                    probe_embedding  = np.stack(probe_data_frame['feature'],axis=0)
                    # probe_embedding  = probe_embedding / np.linalg.norm(probe_embedding, axis=1)[:, np.newaxis]
                if media in probe_embeddings_nonaggregated:
                    probe_embeddings_nonaggregated[media].append(probe_embedding)
                else:
                    probe_embeddings_nonaggregated[media] = [probe_embedding]
    with open("./data/precomputed_features/evaluate/retina_face_probe_embeddings_nonaggregated.pickle", "wb") as file:
        pickle.dump(probe_embeddings_nonaggregated, file)
    
    with open("./data/precomputed_features/evaluate/retina_face_probe_embeddings_nonaggregated.pickle", "rb") as file:
        probe_embeddings_nonaggregated = pickle.load(file)
    with open("./data/precomputed_features/evaluate/retina_face_gall1_embeddings_nonaggregated.pickle", "rb") as file:
        gall1_embeddings_nonaggregated = pickle.load(file)
    with open("./data/precomputed_features/evaluate/retina_face_gall2_embeddings_nonaggregated.pickle", "rb") as file:
        gall2_embeddings_nonaggregated = pickle.load(file)
        
    print("Total number of gallery identities: ", len(gall1_embeddings_nonaggregated.keys()))
    print("Number of NAN probes media: ", 0, " Total number of probes media: ", len(probe_embeddings_nonaggregated.keys()))        
    print("Total number of gallery identities: ", len(gall2_embeddings_nonaggregated.keys()))

    probe_embeddings_aggregated = {}
    gall_id_1_embeddings_aggregated = {}
    gall_id_2_embeddings_aggregated = {}
    gall_id_1_embeddings_not_idaggregated = {}
    gall_id_2_embeddings_not_idaggregated = {}
    
    print("Average probes to get media level embeddings")
    for media, embeddings_list in tqdm(probe_embeddings_nonaggregated.items()):
        templist = []
        if (probe_aggregate_type == "sub"): # sub means aggregate subfiles and then aggregate again, full means aggregate all at once.
            for embeddings in embeddings_list:
                feat = np.stack(embeddings,axis=0)
                agg_embedding = fuse_function(media, model, type = "probe", featureset = feat, axis=0, keepdims=True, baseline=baseline)
                templist.append(agg_embedding)
        elif (probe_aggregate_type == "full"):
            embeddings    = np.concatenate(embeddings_list, axis=0) # list of all embeddings
            agg_embedding = np.stack(embeddings,axis=0)
            templist.append(agg_embedding)
        agg_embedding = fuse_function(media, model, type = "probe", featureset = np.concatenate(templist, axis=0), axis=0, keepdims=True, baseline=baseline)
        if (len(agg_embedding.shape) == 3):
            agg_embedding = agg_embedding.reshape(agg_embedding.shape[0], agg_embedding.shape[2]*agg_embedding.shape[1])
        probe_embeddings_aggregated[media] = agg_embedding
                
    print("Average gallery 2 to get media level embeddings")
    for id in tqdm(gall2_embeddings_nonaggregated.keys()):
        id_embedding_list = []
        for media, embeddings_list in gall2_embeddings_nonaggregated[id].items():
            if gallery_aggregate_type == "sub":
                templist = []
                for embeddings in embeddings_list:
                    feat          = np.stack(embeddings,axis=0)
                    agg_embedding = fuse_function(media, model, type = "gallery", featureset = feat,axis=0,keepdims=True, baseline=baseline)
                    templist.append(agg_embedding)
                templist = np.concatenate(templist, axis=0)
                gallerymedia_agg_embedding = fuse_function(media, model, type = "gallery", featureset = templist, axis=0, keepdims=True, baseline=baseline)
                id_embedding_list.append(gallerymedia_agg_embedding)
            elif (gallery_aggregate_type == "mid"):
                embeddings        = np.concatenate(embeddings_list, axis=0) # list of all embeddings
                stacked_embedding = np.stack(embeddings,axis=0)
                gallerymedia_agg_embedding = fuse_function(media, model, type = "gallery", featureset = stacked_embedding, axis=0, keepdims=True, baseline=baseline)
                id_embedding_list.append(gallerymedia_agg_embedding)
            elif (gallery_aggregate_type == "full"):
                embeddings        = np.concatenate(embeddings_list, axis=0) # list of all embeddings
                stacked_embedding = np.stack(embeddings,axis=0)
                id_embedding_list.append(stacked_embedding)
        id_level_agg_embedding = fuse_function(media, model, type = "gallery", featureset = np.concatenate(id_embedding_list, axis=0), axis=0, keepdims=True, baseline=baseline)
        # id_level_agg_embedding = id_level_agg_embedding.reshape(id_level_agg_embedding.shape[0], id_level_agg_embedding.shape[2] * id_level_agg_embedding.shape[1])        
        gall_id_2_embeddings_not_idaggregated[id] = np.concatenate(id_embedding_list, axis=0)
        gall_id_2_embeddings_aggregated[id] = id_level_agg_embedding
    print("gallery 2 number of keys: ", len(gall_id_2_embeddings_aggregated.keys()))
    
    print("Average gallery 1 to get media level embeddings")
    for index, id in enumerate(tqdm(gall1_embeddings_nonaggregated.keys())):
        id_embedding_list = []
        media_list        = []
        for media, embeddings_list in gall1_embeddings_nonaggregated[id].items():
            media_list.append(media)
            if gallery_aggregate_type == "sub":
                templist = []
                for embeddings in embeddings_list:
                    feat          = np.stack(embeddings,axis=0)
                    agg_embedding = fuse_function(media, model, type = "gallery", featureset = feat, axis=0, keepdims=True, baseline=baseline)
                    templist.append(agg_embedding)
                templist = np.concatenate(templist, axis=0)
                gallerymedia_agg_embedding = fuse_function(media, model, type = "gallery", featureset = templist, axis=0, keepdims=True, baseline=baseline)
                id_embedding_list.append(gallerymedia_agg_embedding)
            elif (gallery_aggregate_type == "mid"):
                embeddings        = np.concatenate(embeddings_list, axis=0) # list of all embeddings
                stacked_embedding = np.stack(embeddings,axis=0)
                gallerymedia_agg_embedding = fuse_function(media, model, type = "gallery", featureset = stacked_embedding, axis=0, keepdims=True, baseline=baseline)
                id_embedding_list.append(gallerymedia_agg_embedding)
            elif (gallery_aggregate_type == "full"):
                embeddings        = np.concatenate(embeddings_list, axis=0) # list of all embeddings
                stacked_embedding = np.stack(embeddings,axis=0)
                id_embedding_list.append(stacked_embedding)
       
        # with open("media_list.txt", "w+") as file:
        #     for item in media_list:
        #         file.write(f'{item}\n')
                
        id_level_agg_embedding = fuse_function(id, model, type = "gallery", featureset = np.concatenate(id_embedding_list, axis=0), axis=0, keepdims=True, baseline=baseline)
        # id_level_agg_embedding = id_level_agg_embedding.reshape(id_level_agg_embedding.shape[0], id_level_agg_embedding.shape[2] * id_level_agg_embedding.shape[1])
        gall_id_1_embeddings_not_idaggregated[id] = np.concatenate(id_embedding_list, axis=0)
        gall_id_1_embeddings_aggregated[id] = id_level_agg_embedding
    print("gallery 1 number of keys: ", len(gall_id_1_embeddings_aggregated.keys()))

    gallery_1_id_order = list(gall_id_1_embeddings_aggregated.keys())
    gallery_2_id_order = list(gall_id_2_embeddings_aggregated.keys())
    probes_media_order = list(probe_embeddings_aggregated.keys())

    with open('gallery1_embeddings.pickle', 'wb') as handle:
        pickle.dump(gall_id_1_embeddings_aggregated, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('gallery2_embeddings.pickle', 'wb') as handle:
        pickle.dump(gall_id_2_embeddings_aggregated, handle, protocol=pickle.HIGHEST_PROTOCOL)

    gallery_featues_1 = np.concatenate(list(gall_id_1_embeddings_aggregated.values()),axis=0)
    gallery_featues_2 = np.concatenate(list(gall_id_2_embeddings_aggregated.values()),axis=0)
    probe_embeddings  = np.concatenate(list(probe_embeddings_aggregated.values()),axis=0)   
   
    gallery_featues_1 = normalizeRows(gallery_featues_1)
    gallery_featues_2 = normalizeRows(gallery_featues_2)
    probe_features    = normalizeRows(probe_embeddings)

    score_matrix_1    = np.dot(probe_features,gallery_featues_1.T)
    score_matrix_2    = np.dot(probe_features,gallery_featues_2.T)
    score_matrix      = np.concatenate([score_matrix_1,score_matrix_2],axis=1)

    print(score_matrix_1.shape, score_matrix_2.shape, score_matrix.shape)
    
    probedf    = pd.DataFrame()
    gallerydf1 = pd.DataFrame()
    gallerydf2 = pd.DataFrame()

    probe_SUBID = []
    for each in probes_media_order:
        probe_SUBID.append(each.split('_')[1])
    probedf['subject_id']    = probe_SUBID        # probe subid order
    gallerydf1['subject_id'] = gallery_1_id_order # gallery1_id_order
    gallerydf2['subject_id'] = gallery_2_id_order # gallery2_id_order

    gallerydf  = pd.concat([gallerydf1,gallerydf2])
    FACE_COLOR = 'red'
    alg_label  = 'all'

    if (baseline != True):
        score_matrix = scipy.special.softmax(score_matrix, axis=1)

    if baseline == True:
        face_verification_all = VerificationResults(score_matrix,probedf,gallerydf,algorithm=alg_label+"-Face",label='Baseline all Probes',color='red')
    else:
        face_verification_all = VerificationResults(score_matrix,probedf,gallerydf,algorithm=alg_label+"-Face",label='ProxyFusion all Probes',color='blue')

    roc_frame,_    = face_verification_all.createOldRoc()

    plt.figure(figsize=LARGE_FIGURE)
    ax = plt.subplot(1,1,1)
    ax.set_title("Receiver Operating Characteristic: ")
    ax.set_xlim(1e-5,1.0)
    ax.set_ylim(-0.05,1.05)
    ax.set_xlabel('False Accept Rate')
    ax.set_ylabel('True Accept Rate')
    ax.set_xscale('log')   
 
    out = metrics_at_thersholds(roc_frame)
    print(out)
    return out

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train ProxyFusion model.')
    parser.add_argument('--selected_experts', type=int, required=True, help='Number of K selected experts')
    parser.add_argument('--total_experts', type=int, required=True, help='Number of total number of proxies/experts')
    parser.add_argument('--proxy_loss_weightage', type=float, default = 0.01, help='weightage for proxy loss')
    parser.add_argument('--feature_dim', type=int, required=True, help='Original dimensionality of precomputed features, and proxies initial dim')
    parser.add_argument('--domain_dim', type=int, required=True, help='projection dimensionality from proxies/features')
    parser.add_argument('--subjects_per_batch', type=int, default = 170, help='projection dimensionality from proxies/features')
    parser.add_argument('--subject_repeat_factor', type=int, default = 2, help='projection dimensionality from proxies/features')
    parser.add_argument('--num_workers', type=int, required=True, help='Number of data loader workers')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--pretrained_checkpoint_path', type=str, required=True, help='Path to pretrained checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--bts_protocol_path', type=str, required=True, help='Path to CSV with BTS3.1 probes, gallery 1 and gallery 2')
    parser.add_argument('--bts_embeddings_path', type=str, required=True, help='Path to probes, gallery 1 and gallery 2 precomputed embeddings')
    parser.add_argument('--face_detector', type=str, required=True, help='Retinaface / MTCNN')
    parser.add_argument('--face_feature_extractor', type=str, required=True, help='Adaface / Arcface')

    args = parser.parse_args()

    model = ProxyFusion(DIM=512)
    model = model.cuda()    
    checkpoint = torch.load(args.pretrained_checkpoint_path)
    model.load_state_dict(checkpoint['state_dict']["model_weights"])
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    plt.figure(figsize=LARGE_FIGURE)
    ax = plt.subplot(1,1,1)
    ax.set_title("Receiver Operating Characteristic: Overview ")
    ax.set_xlim(1e-5,1.0)
    ax.set_ylim(-0.05,1.05)
    ax.set_xlabel('False Accept Rate')
    ax.set_ylabel('True Accept Rate')
    ax.set_xscale('log')

    print("======== GAP (Global Average Pooling) Results ==========")    
    print (Evaluate_Model(model,baseline=True,ax=ax,linestyle='dashed',color='red', probe_aggregate_type = "full", gallery_aggregate_type = "mid"))
    print("======== ProxyFusion Results ==========")
    print (Evaluate_Model(model,baseline=False,ax=ax,linestyle='dashed',color='red', probe_aggregate_type = "full", gallery_aggregate_type = "mid"))
