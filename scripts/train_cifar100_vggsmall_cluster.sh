#!/usr/bin/env bash

python examples/classifier_cifar100/main.py ~/datasets/data.cifar100 \
    -a cifar100_vggsmall_cluster_q -j 10 -b 128 -p 20 --epochs 100 --qw 3 \
    --gpu $1 --log-name $2 --lr 1e-4 --cosine --debug --pretrained
