#!/usr/bin/env bash

# This script runs a Tensorflow Object Detection training job. Make sure to 
# run from the tf1.12-gpu conda environment and to be in the tensorflow/
# models/research directory.

#export CUDA_VISIBLE_DEVICES=0,1
PIPELINE_CONFIG_PATH=/nfs0/BPP/Fowler_Lab/warman/computer_vision/EarVision/models/2019_training/models/faster_rcnn_inception_resnet_v2_atrous_coco_v100_train.config
MODEL_DIR=/nfs0/BPP/Fowler_Lab/warman/computer_vision/EarVision/models/2019_training/models/model/
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py \
	--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
	--model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
	--alsologtostderr
