#!/usr/bin/env bash
python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
    -a cifar10_vggsmall_lsq -j 10 -b 256 --pretrained --warmup-epoch -1 \
    --gpus $1 --log-name $2 --lr 0.0002 --epochs 100 --lr-scheduler CosineAnnealingLR \
    --qw 4 --qa 4 --q-mode layer_wise --debug


#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall_q -j 10 -b 256 --pretrained \
#    --gpu $1 --log-name $2 --lr 0.0002 --epochs 100 --cosine --l1

# 93.18