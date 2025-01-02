# ProxyFusion

This is the codebase for ProxyFusion: Face Feature Aggregation Through Sparse Experts

> Abstract:
Face feature fusion is indispensable for robust face recognition, particularly in scenarios involving long-range, low-resolution media (unconstrained environments) where not all frames or features are equally informative. Existing methods often rely on large intermediate feature maps or face metadata information, making them incompatible with legacy biometric template databases that store pre-computed features. Additionally, real-time inference and generalization to large probe sets remains challenging. To address these limitations, we introduce a linear time proxy based sparse expert selection and pooling approach for context driven feature-set attention. Our approach is order invariant on the feature-set, generalizes to large sets, is compatible with legacy template stores, and utilizes significantly less parameters making it suitable real-time inference and edge use-cases. Through qualitative experiments, we demonstrate that ProxyFusion learns discriminative information for importance weighting of face features without relying on intermediate features. Quantitative evaluations on challenging low-resolution face verification datasets such as IARPA BTS3.1 and DroneSURF show the superiority of ProxyFusion in unconstrained long-range face recognition setting. Code and pretrained models will be released upon acceptance.

<img src="./images/architecture.png" alt="architecture" width="1000"/>

## Setup Environment

```
conda env create -f environment.yml
conda activate reid_briar_env

or 

conda env create -f environment.yml -n new_env_name
```

## Data

#### BRIAR
We utilize the BRIAR - BRS1, 2 and 3 datasets for training our model. Kindly reach to authors of [Expanding Accurate Person Recognition to New Altitudes and Ranges: The
BRIAR Dataset](https://openaccess.thecvf.com/content/WACV2023W/LRR/papers/Cornett_Expanding_Accurate_Person_Recognition_to_New_Altitudes_and_Ranges_The_WACVW_2023_paper.pdf) for access to the dataset. We utilize the associated BTS3.1 dataset and its respective protocol for evaluation.

#### DroneSURF

We also present results on DroneSURF dataset. To request droneSURF dataset visit: [https://iab-rubric.org/index.php/dronesurf](DroneSURF) or reach out to the authors of [DroneSURF: Benchmark Dataset for Drone-based Face Recognition
](https://ieeexplore.ieee.org/document/8756593)


### Precompute Embeddings:

Follow instructions at https://github.com/mk-minchul/AdaFace to install Adaface and precompute embeddings. For Arcface, follow the instructions at https://github.com/ronghuaiyang/arcface-pytorch and utilize the pretrained checkpoint to compute the Arcface embeddings. Note that by default Adaface and Arcface repositories use MTCNN face detector. For retinaface, follow instructions at https://github.com/serengil/retinaface to crop and align faces using Retinaface, then pass them through Adaface and Arcface. 

### Data Folder structure:
For fast dataloading for model training, we utilize the HDF5 files. To create HDF5 Files from precomputed embeddings:

```
python3 ./utils/create_hdf5.py
```

The final folder structure for training and evaluation dataset should look like:
```
- data
    - train
    - MTCNN
        - Adaface
            - features_G03001.hdf5
            - features_G03002.hdf5
            ...
        - Arcface
            - features_G03001.hdf5
            - features_G03002.hdf5
            ...
    - Retinaface
        - Adaface
            - features_G03001.hdf5
            - features_G03002.hdf5
            ...
        - Arcface
            - features_G03001.hdf5
            - features_G03002.hdf5
            ...
```

Each hdf5 file has the following structure:
```
G03377/gallery/images/98
G03377/gallery/images/99
G03377/gallery/video
G03377/gallery/video/0
G03377/gallery/video/1
G03377/gallery/video/10
G03377/probe_61
G03377/probe_62
G03377/probe_63
```
where each element is `<subject_id>/<probe_probeindex>` for probes and `<subject_id>/<gallery>/<images/videos>/index`

## Training:

Run the below command to run the training on BRIAR Dataset:

```
python train.py \
    --selected_experts 4 \
    --total_experts 4 \
    --proxy_loss_weightage 0.01 \
    --feature_dim 128 \
    --domain_dim 10 \
    --subjects_per_batch 170 \
    --subject_repeat_factor 2 \
    --num_workers 4 \
    --num_epochs 700 \
    --Checkpoint_Saving_Path "./logs" \
    --data_path "/path/to/training_hdf5_embeddings" \
    --bts_protocol_path "/path/to/bts_protocol.csv" \
    --bts_embeddings_path "/path/to/bts_embeddings" \
    --face_detector "Retinaface" \
    --face_feature_extractor "Adaface"
```

## Evaluation and pretrained checkpoints:

Evaluation scripts can be found in `./evaluation/` folder.  Pretrained checkpoints are available  To evaluate, for instance, Run the following command to the evaluation on BTS3.1 dataset.

Pretrained checkpoints for the models are available at: https://drive.google.com/drive/folders/13XvZXnJyxakFBdR22TTP5x_QdmmObScz?usp=sharing

```
python ./evaluation/Eval_BRIAR.py \
    --selected_experts 4 \
    --total_experts 4 \
    --proxy_loss_weightage 0.01 \
    --feature_dim 128 \
    --domain_dim 10 \
    --subjects_per_batch 170 \
    --subject_repeat_factor 2 \
    --num_workers 4 \
    --num_epochs 700 \
    --pretrained_checkpoint_path "./checkpoints/<pretrained_ckpts>" \
    --data_path "/path/to/training_hdf5_embeddings" \
    --bts_protocol_path "/path/to/bts_protocol.csv" \
    --bts_embeddings_path "/path/to/bts_embeddings" \
    --face_detector "Retinaface" \
    --face_feature_extractor "Adaface"
```

## Runs 

**Training:**
![Training Screenshot](./images/training_screenshot.png)
**Evaluation:**
![Evaluation Screenshot](./images/evaluation_screenshot.png)

### Bibliography / References:

```
@inproceedings{
kim2022cluster,
title={Cluster and Aggregate: Face Recognition with Large Probe Set},
author={Minchul Kim and Feng Liu and Anil Jain and Xiaoming Liu},
booktitle={Advances in Neural Information Processing Systems, NeurIPS, 2022},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=5yAmUvdXAve}
}
```


```
@INPROCEEDINGS{9880230,
  author={Kim, Minchul and Jain, Anil K. and Liu, Xiaoming},
  booktitle={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={AdaFace: Quality Adaptive Margin for Face Recognition}, 
  year={2022},
  volume={},
  number={},
  pages={18729-18738},
  keywords={Image quality;Training;Computer vision;Adaptation models;Codes;Face recognition;Training data;Face and gestures; Recognition: detection;categorization;retrieval},
  doi={10.1109/CVPR52688.2022.01819}}
```

```
@article{deng2019retinaface,
  title={Retinaface: Single-stage dense face localisation in the wild},
  author={Deng, Jiankang and Guo, Jia and Zhou, Yuxiang and Yu, Jinke and Kotsia, Irene and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:1905.00641},
  year={2019}
}
```
