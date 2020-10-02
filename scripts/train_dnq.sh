#!/usr/bin/env bash
# AlexNet baseline
#python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
#    -a alexnet -j 10 -b 256 --log-name $2 \
#    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --epochs 90 --lr-scheduler StepLR --step-size 30


#python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
#    -a alexnet_dnq -j 10 -b 256 --log-name $2 \
#    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --epochs 90 --lr-scheduler StepLR --step-size 30 \
#    --qw 4 --qa 4 --q-mode layer_wise --debug \
#    --dnq-scheduler MultiStepDNQ --dnq-milestones 25 55 --dnq-gamma 2


#python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 256 --log-name $2 \
#    --lr 0.1 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --epochs 90 --lr-scheduler StepLR --step-size 30