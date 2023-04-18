#!/usr/bin/env bash

# Clone the lsun repo to get scripts to download the dataset
git clone https://github.com/fyu/lsun.git
cd lsun/

# Download dataset
python download.py -c bedroom -o ../

cd ..
rm -rf lsun

# Unzip files
unzip bedroom_train_lmdb.zip -d train
unzip bedroom_val_lmdb.zip -d val

rm bedroom_train_lmdb.zip
rm bedroom_val_lmdb.zip

python /notebooks/consistency_models/datasets/lsun_bedroom.py \
    /notebooks/datasets/lsun_bedroom/train/bedroom_train_lmdb/ \
    /notebooks/datasets/lsun_bedroom/train/processed/
