#!/usr/bin/env bash

#python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 256 --log-name $2 \
#    --lr 0.01 --gpu $1 --epochs 90

python examples/classifier_imagenet/main.py \
    --dist-url 'tcp://127.0.0.1:9090' --dist-backend 'nccl' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    ~/datasets/data.imagenet \
    -a resnet18 --lr 0.01 -j 10 -b 256 --epochs 90 \
    --log-name $1

