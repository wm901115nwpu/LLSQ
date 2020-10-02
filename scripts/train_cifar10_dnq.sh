#!/usr/bin/env bash
# Decreasing network quantization for efficient training
#
#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall_dnq -j 10 -b 100 -p 20 --epochs 400 \
#    --gpus $1 --log-name $2 --lr 0.02 --qa 16 --qw 16 --q-mode layer_wise \
#    --dnq-scheduler MultiStepDNQ --dnq-milestones 400 --dnq-gamma 2 \
#    --lr-scheduler MultiStepLR --milestones 80 160 300 \
#    --debug

python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
    -a cifar10_vggsmall_dnqv2 -j 10 -b 100 -p 20 --epochs 400 \
    --gpu $1 --log-name $2 --lr 0.02 --qa 16 --qw 16 --q-mode layer_wise \
    --dnq-scheduler MultiStepDNQ --dnq-milestones 400 --dnq-gamma 2 \
    --lr-scheduler MultiStepLR --milestones 80 160 300 \
    --debug

# 0.002 ==warm-up==> [(1, 0.02), (80, 0.002), (160, 0.0002), (300, 0.00002)]
#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall -j 10 -b 100 -p 20 --wd 5e-4 --epochs 400 \
#    --lr-scheduler MultiStepLR --milestones 80 160 300 \
#    --gpus $1 --log-name $2 --lr 0.02