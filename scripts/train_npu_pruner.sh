#!/usr/bin/env bash
# resume and test
#python examples/classifier_imagenet/main_npu.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --debug --non-zero-num 28 -e

# straight NPU 7
#python examples/classifier_imagenet/main_npu.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --debug --non-zero-num 7

# INS NPU 7
#python examples/classifier_imagenet/main_npu.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --debug --non-zero-num 7 --INS --beta 0.1

# NPU 7 ADMM step 1 multi gpu (4)
#python examples/classifier_imagenet/main_npu.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 2048 --pretrained \
#    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
#    --log-name $1 --epochs 180 --lr-scheduler CosineAnnealingLR \
#    --debug --non-zero-num 7 --admm --rho 0.01 --rho-dynamic \
#    --dist-url 'tcp://127.0.0.1:9091' --dist-backend 'nccl' \
#    --multiprocessing-distributed --world-size 1 --rank 0

# # NPU 7 ADMM step 2
#python examples/classifier_imagenet/main_npu.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 40 --lr-scheduler CosineAnnealingLR \
#    --debug --non-zero-num 7 --resume-after $3




# Level Pruner ADMM step 1
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --debug --sparsity 0.95 --admm --rho 0.01 --rho-dynamic
# Level Pruner ADMM step 2
#python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
#    -a resnet18 -j 10 -b 512 --pretrained \
#    --lr 0.001 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 10 --lr-scheduler CosineAnnealingLR \
#    --debug --sparsity 0.95 --resume-after $3


#==========cifar10===========
# straight pruning
#python examples/classifier_cifar10/main_admm.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall -j 10 -b 128 --pretrained \
#    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --debug --sparsity 0.95

# admm step 1
#python examples/classifier_cifar10/main_admm.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall -j 10 -b 128 --pretrained \
#    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --debug --admm --sparsity 0.95 --rho 0.05 --rho-dynamic
# admm step 2
#python examples/classifier_cifar10/main_admm.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall -j 10 -b 128 --pretrained \
#    --lr 0.001 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 5 --lr-scheduler CosineAnnealingLR \
#    --debug --sparsity 0.95 --resume-after $3

# straight npu pruning
#python examples/classifier_cifar10/main_admm_npu.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall -j 10 -b 128 --pretrained \
#    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --debug --non-zero-num 7

# INS npu pruning
#python examples/classifier_cifar10/main_admm_npu.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall -j 10 -b 128 --pretrained \
#    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --debug --non-zero-num 7 --INS --beta 0.1

# admm step 1 npu pruning
#python examples/classifier_cifar10/main_admm_npu.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall -j 10 -b 128 --pretrained \
#    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
#    --debug --admm --non-zero-num 7 --rho 0.05 --rho-dynamic-v2
# admm step 2 npu pruning
#python examples/classifier_cifar10/main_admm_npu.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall -j 10 -b 128 --pretrained \
#    --lr 0.001 --wd 1e-4 --warmup-epoch -1 \
#    --gpu $1 --log-name $2 --epochs 10 --lr-scheduler CosineAnnealingLR \
#    --debug --non-zero-num 7 --resume-after $3