#!/bin/bash
# Define variables
K_g=4
K_p=4
proxy_loss_weightage=0.01
num_workers=4
num_epochs=700
Checkpoint_Saving_Path="./logs"
data_path="/path/to/training_hdf5_embeddings"
bts_protocol_path="/path/to/bts_protocol.csv"
bts_embeddings_path="/path/to/bts_embeddings"
face_detector="Retinaface"
face_feature_extractor="Adaface"

# Run the training script with the defined variables
python train.py \
    --K_g $K_g \
    --K_p $K_p \
    --proxy_loss_weightage $proxy_loss_weightage \
    --num_workers $num_workers \
    --num_epochs $num_epochs \
    --Checkpoint_Saving_Path $Checkpoint_Saving_Path \
    --data_path $data_path \
    --bts_protocol_path $bts_protocol_path \
    --bts_embeddings_path $bts_embeddings_path \
    --face_detector $face_detector \
    --face_feature_extractor $face_feature_extractor
