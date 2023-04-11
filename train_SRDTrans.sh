#!/bin/bash


## train

## 1. Data preparation 
#  Download the demo data(.tif file) and put the noisy data and the clean data into datasets/noisy/ and  datasets/clean/, respectively.


## 2. Train
# Simulated STORM & Simulated Calcium imaging data at ~30hz
##################################################################################################################################
python -u train.py --datasets_folder noisy --datasets_path datasets/ --n_epochs 30 --GPU 0 --train_datasets_size 6000  --patch_x 128 --patch_t 128 --clean_img_path datasets/clean/clean.tif


# Simulated Calcium imaging data at ~0.3hz
##################################################################################################################################
python -u train.py --datasets_folder noisy --datasets_path datasets/ --n_epochs 30 --GPU 0 --train_datasets_size 3000  --patch_x 128 --patch_t 128 --clean_img_path datasets/clean/clean.tif

