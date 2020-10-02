#!/usr/bin/env bash

#python examples/classifier_cifar10/ttq_main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall_ttq -j 10 -b 128 -p 20 --epochs 100 \
#    --gpu $1 --log-name $2 --lr 1e-4 --cosine --debug --pretrained


python examples/classifier_cifar10/ttq_main.py ~/datasets/data.cifar10 \
    -a cifar10_resnet18_tbq -j 10 -b 128 -p 20 --epochs 300 \
    --gpu $1 --log-name $2 --lr 6e-4 --pretrained --cosine


