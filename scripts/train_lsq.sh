#!/usr/bin/env bash
python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
    -a alexnet_lsq -j 10 --pretrained -b 2048 --log-name $2 \
    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
    --gpu $1 --epochs 90 --lr-scheduler CosineAnnealingLR \
    --qw 4 --qa 4 --q-mode layer_wise \
    --debug

#python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
#    -a alexnet_lsq -j 10 --pretrained -b 2048 --log-name $2 \
#    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --qw 2 --qa 2 --q-mode layer_wise \
#    --debug --resume $3

#python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
#    -a resnet18_lsq -j 10 -b 512 --pretrained \
#    --lr 0.01 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --qw 3 --qa 3 --q-mode layer_wise \
#    --debug

#python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
#    -a resnet18_lsq -j 10 -b 512 --pretrained \
#    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --qw 2 --qa 2 --q-mode layer_wise \
#    --debug --resume $3


## LSQF
#python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
#    -a resnet18_f_lsq --gen-map --original-model resnet18
#python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
#    -a resnet18_f_lsq -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 1 --lr-scheduler CosineAnnealingLR \
#    --qw 8 --qa 8 --q-mode layer_wise \
#    --debug
