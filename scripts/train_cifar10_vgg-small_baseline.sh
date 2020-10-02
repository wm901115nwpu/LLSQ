#!/usr/bin/env bash
# In LQ-Net Learning rate is configured as follows:
# [(1, 0.02), (80, 0.002), (160, 0.0002), (300, 0.00002)]) max epoch: 400
# In this project, learning rate is configured as follows:
# 0.002 ==warm-up==> 0.02 ==cosine==> 0 don't work
# 0.002 ==warm-up==> [(1, 0.02), (80, 0.002), (160, 0.0002), (300, 0.00002)]
#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall -j 10 -b 100 -p 20 --epochs 400 \
#    --gpu $1 --log-name $2 --lr 0.02 --lr-scheduler 'MultiStepLR' --milestones 80 160 300 --debug \
#    --warmup-epoch 10
# 93.14

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    --dist-url 'tcp://127.0.0.1:9081' --dist-backend 'nccl' \
#    --multiprocessing-distributed --world-size 1 --rank 0 \
#    -a cifar10_vggsmall -j 10 -b 100 -p 20 --epochs 400 \
#    --log-name $1 --lr 0.02 --lr-scheduler 'MultiStepLR'  --milestones 80 160 300 --debug \
#    --warmup-epoch 10

#python examples/classifier_cifar10/bp_main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall_bp -j 10 -b 128 -p 20 --epochs 400 \
#    --gpu $1 --log-name $2 --lr 0.002 --cosine --debug

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggtiny -j 10 -b 128 -p 20 --epochs 300 \
#    --gpu $1 --log-name $2 --lr 0.002 --cosine --debug

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggtiny -j 10 -b 128 -p 20 --epochs 300 \
#    --gpu $1 --lr 0.002 --cosine -e --pretrained

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall -j 10 -b 128 -p 20 --epochs 400 \
#    --gpu $1 --log-name $2 --lr 0.002
# 92.73

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall -j 10 -b 128 -p 20 --epochs 300 \
#    --gpu $1 --log-name $2 --lr 0.002 --cosine
# 92.44
#VGG(
#  (features): Sequential(
#    (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#    (2): ReLU(inplace)
#    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#    (5): ReLU(inplace)
#    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#    (7): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#    (8): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#    (9): ReLU(inplace)
#    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#    (11): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#    (12): ReLU(inplace)
#    (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#    (14): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#    (15): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#    (16): ReLU(inplace)
#    (17): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#    (18): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#    (19): ReLU(inplace)
#    (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#  )
#  (classifier): Linear(in_features=8192, out_features=10, bias=True)
#)
