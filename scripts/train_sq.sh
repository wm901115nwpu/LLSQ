#!/usr/bin/env bash
# sparsity => quantization
# resnet18 w0a0s0.3
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 10 --lr-scheduler CosineAnnealingLR \
#    --qw -3 --qa -3 --sparsity 0.3 \
#    --debug

# resnet18 w0a0s0.4
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 20 --lr-scheduler CosineAnnealingLR \
#    --qw -3 --qa -3 --sparsity 0.4 \
#    --debug

# resnet18 w0a0s0.5
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 50 --lr-scheduler CosineAnnealingLR \
#    --qw -3 --qa -3 --sparsity 0.5 --INS --beta 0.1 \
#    --debug

# resnet18 w0a0s0.6
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.005 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 60 --lr-scheduler CosineAnnealingLR \
#    --qw -3 --qa -3 --sparsity 0.6 --INS --beta 0.1 \
#    --debug

# resnet18 w0a0s0.65
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.005 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 60 --lr-scheduler CosineAnnealingLR \
#    --qw -3 --qa -3 --sparsity 0.65 --INS --beta 0.1 \
#    --debug

# resnet18 w0a0s0.7
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --qw -3 --qa -3 --sparsity 0.7 --INS --beta 0.1 \
#    --debug

# resnet18 w0a0s0.75
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --qw -3 --qa -3 --sparsity 0.75 --INS --beta 0.1 \
#    --debug

# resnet18 w0a0s0.4>w8a8s0.4
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 3 --lr-scheduler CosineAnnealingLR \
#    --qw 8 --qa 8 --sparsity 0.4 --resume-after $3 \
#    --debug

# resnet18 w0a0s0.4>w6a6s0.4
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 30 --lr-scheduler CosineAnnealingLR \
#    --qw 6 --qa 6 --sparsity 0.4 --resume-after $3 \
#    --debug

# resnet18 w0a0s0.4>w5a5s0.4
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.005 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 60 --lr-scheduler CosineAnnealingLR \
#    --qw 5 --qa 5 --sparsity 0.4 --resume-after $3 \
#    --debug

# resnet18 w0a0s0.4>w4a4s0.4
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.01 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --qw 4 --qa 4 --sparsity 0.4 --resume-after $3 \
#    --debug

# resnet18 w0a0s0.65>w4a4s0.65
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.01 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --qw 4 --qa 4 --sparsity 0.65 --resume-after $3 \
#    --debug

# resnet18 w0a0s0.65>w3a3s0.65
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.01 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --qw 3 --qa 3 --sparsity 0.65 --resume-after $3 \
#    --debug

# resnet18 w0a0s0.7>w4a4s0.7
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.01 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --qw 4 --qa 4 --sparsity 0.7 --resume-after $3 \
#    --debug

# resnet18 w0a0s0.7>w3a3s0.7
python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
    -a resnet18 -j 10 -b 512 --pretrained \
    --lr 0.01 --wd 5e-5 --warmup-epoch -1 \
    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
    --qw 0 --qa 0 --sparsity 0.85 --resume-after $3 \
    --debug -e

# resnet18 w3a3s0.75 one-step
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.01 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 120 --lr-scheduler CosineAnnealingLR \
#    --qw 3 --qa 3 --sparsity 0.75 --INS --beta 0.1 \
#    --debug

# evaluation
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet50 -j 10 -b 256 --pretrained \
#    --lr 0.01 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --qw 0 --qa 0 --sparsity 0.90 --INS --beta 0.1 \
#    --debug

#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.01 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --qw 0 --qa 0 --sparsity 0.85 --INS --beta 0.1 \
#    --debug

#Acc@1 77.684 Acc@5 93.594
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a efficientnet_b0 -j 10 -b 128 --pretrained \
#    --lr 0.01 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --qw 0 --qa 0 --sparsity 0.6 --INS --beta 0.1 \
#    --debug

#==============================================================================
# quantization => sparsity

# resnet18 w8a8s0.0
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 5 --lr-scheduler CosineAnnealingLR \
#    --qw 8 --qa 8 --sparsity 0.0 --INS --beta 0.1 \
#    --debug -e

# resnet18 w6a6s0.0 epoch 10
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 30 --lr-scheduler CosineAnnealingLR \
#    --qw 6 --qa 6 --sparsity 0.0 --INS --beta 0.1 \
#    --debug

# resnet18 w5a5s0.0
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.005 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 60 --lr-scheduler CosineAnnealingLR \
#    --qw 5 --qa 5 --sparsity 0.0 --INS --beta 0.1 \
#    --debug

# resnet18 w4a4s0.0
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.01 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --qw 4 --qa 4 --sparsity 0.0 --INS --beta 0.1 \
#    --debug

# resnet18 w3a3s0.0
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.01 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --qw 3 --qa 3 --sparsity 0.0 --INS --beta 0.1 \
#    --debug


# resnet18 w6a6s0.0=w6a6s0.3
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 5 --lr-scheduler CosineAnnealingLR \
#    --qw 6 --qa 6 --sparsity 0.3 --resume-after $3 \
#    --debug

# resnet18 w6a6s0.0=w6a6s0.4
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 20 --lr-scheduler CosineAnnealingLR \
#    --qw 6 --qa 6 --sparsity 0.4 --resume-after $3 \
#    --debug

# resnet18 w6a6s0.0=w6a6s0.5
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 50 --lr-scheduler CosineAnnealingLR \
#    --qw 6 --qa 6 --sparsity 0.5 --INS --beta 0.1 --resume-after $3 \
#    --debug

# resnet18 w6a6s0.0=w6a6s0.6
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 60 --lr-scheduler CosineAnnealingLR \
#    --qw 6 --qa 6 --sparsity 0.6 --INS --beta 0.1 --resume-after $3 \
#    --debug

# resnet18 w6a6s0.0=w6a6s0.65
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 60 --lr-scheduler CosineAnnealingLR \
#    --qw 6 --qa 6 --sparsity 0.65 --INS --beta 0.1 --resume-after $3 \
#    --debug

# resnet18 w6a6s0.0=w6a6s0.7
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.01 --wd 5e-5 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --qw 6 --qa 6 --sparsity 0.7 --INS --beta 0.1 --resume-after $3 \
#    --debug