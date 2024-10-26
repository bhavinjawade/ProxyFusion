import  os
from    re import sub
import  random
from    sklearn import metrics
from    torch.optim.lr_scheduler import ExponentialLR, StepLR, MultiStepLR, CosineAnnealingLR
from    torch.utils.data import Dataset, DataLoader
from    torchvision import transforms
from    torchvision.transforms import functional as F
import  numpy as np
import  torch
from    PIL import Image
from    tqdm import tqdm
import  cv2
import  pandas as pd
from    PIL import Image
from    models.fusion_models import ProxyFusion
from    losses.criterion import ProxyConcat_Loss, LatticeLoss
import  pickle
from    tqdm import tqdm
import  glob
from    evaluation.Eval_BRIAR import Evaluate_Model
import  matplotlib
import  matplotlib.pyplot as plt
import  warnings
from    transformers import Adafactor
from    itertools import chain
from    datasets.BRS_Dataset import BRSDataset
import  argparse

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

# Argument parser
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
parser.add_argument('--Checkpoint_Saving_Path', type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
parser.add_argument('--bts_protocol_path', type=str, required=True, help='Path to CSV with BTS3.1 probes, gallery 1 and gallery 2')
parser.add_argument('--bts_embeddings_path', type=str, required=True, help='Path to probes, gallery 1 and gallery 2 precomputed embeddings')
parser.add_argument('--face_detector', type=str, required=True, help='Retinaface / MTCNN')
parser.add_argument('--face_feature_extractor', type=str, required=True, help='Adaface / Arcface')

args = parser.parse_args()

# Define Model
model           = ProxyFusion(DIM=args.feature_dim)

model.K_g        = args.selected_experts
model.K_p        = args.selected_experts
model.K_g_all    = args.total_experts
model.K_p_all    = args.total_experts
model.domain_dim = args.domain_dim

model           = torch.nn.DataParallel(model)
model           = model.cuda()

criterion       = ProxyConcat_Loss(model.module.K_g,model.module.K_p).cuda()
proxy_loss      = LatticeLoss(model.module.K_g_all, model.module.domain_dim).cuda()

combined_params = chain(model.module.parameters(), criterion.parameters())
optimizer       = Adafactor(combined_params, scale_parameter=True, relative_step=True)

# Evaluate Baseline (GAP) before training
print("========================== GAP Performance ===========================")
metrics = Evaluate_Model(model, args, baseline=True,linestyle='dashed',color='red', probe_aggregate_type = "full", gallery_aggregate_type = "mid")
print(metrics, sum(metrics))

NUM_EPOCHS              = args.num_epochs
torch.cuda.empty_cache()

data_path               = args.data_path + '/' + args.face_detector + '/' + args.face_feature_extractor

subjects = [f.split("_")[-1].split(".")[0] for f in os.listdir(data_path) if f.endswith('.hdf5')]
print(len(subjects))

train_dataset = BRSDataset(args, data_path, subjects)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=1,num_workers=4,shuffle=True,pin_memory=True, drop_last=False)

best_tar_sum = float('-inf')
for epoch in range(NUM_EPOCHS):
    model.train()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        probes, gallery, probe_lengths, gallery_lengths, probes_target, gallery_target = batch
        agg_probe, proxies_p, agg_gallery, proxies_g = model(probes, gallery, probe_lengths, gallery_lengths)
        identity_loss   = criterion(agg_probe, agg_gallery, probes_target, gallery_target)
        proxy_loss      = proxy_loss(proxies_p) + proxy_loss(proxies_g)
        loss            = identity_loss + args.proxy_loss_weightage * proxy_loss
        print ("Epoch: ", epoch," | Loss: ", loss)
        loss.backward()
        optimizer.step()
    # scheduler.step()
    
    if ((epoch + 1) in [400, 600]):
        print("Dropped Proxies Learning Rate")
        optimizer.param_groups[4]['lr'] = 0.005
        optimizer.param_groups[5]['lr'] = 0.005

    if ((epoch+1) % 2 == 0):
        model.eval()
        with torch.no_grad():
            print("========================== Epoch " + str(epoch) + " ===========================")
            metrics = Evaluate_Model(model, args, model.module.K_g,model.module.K_p,baseline=False,linestyle='dashed',color='red', probe_aggregate_type = "full", gallery_aggregate_type = "mid")
            print(metrics, sum(metrics))
            state_dict = {"model_weights": model.module.state_dict(), "criterion_parameters": criterion.state_dict()}
            optimizer_dict = optimizer.state_dict()
            torch.save({
                'state_dict': state_dict,
                'optimizer_dict': optimizer_dict,
                'epoch': (epoch+1),
            }, os.path.join(args.Checkpoint_Saving_Path, feature_extractor + '_RetinaFace_Proxy_' + str(epoch + 1) + '_' + str(metrics[1])[0:6] + '.pth.tar'))
    else:
        state_dict = {"model_weights": model.module.state_dict(), "criterion_parameters": criterion.state_dict()}
        optimizer_dict = optimizer.state_dict()
        torch.save({
            'state_dict': state_dict,
            'optimizer_dict': optimizer_dict,
            'epoch': (epoch+1),
            }, os.path.join(args.Checkpoint_Saving_Path, feature_extractor + '_RetinaFace_Proxy_' + str(epoch + 1) + '_' + str(metrics[1])[0:6] + '.pth.tar'))
