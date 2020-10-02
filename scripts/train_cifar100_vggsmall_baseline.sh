#!/usr/bin/env bash
#python examples/classifier_cifar100/main.py ~/datasets/data.cifar100 \
#    -a cifar100_vggsmall -j 10 -b 128 -p 20 --epochs 200 \
#    --gpu $1 --log-name $2 --lr 0.01 --wd 5e-4 \
#    --warmup-epoch 10

#python examples/classifier_cifar100/main.py ~/datasets/data.cifar100 \
#    -a cifar100_vggsmall -j 10 -b 128 -p 20 --epochs 200 \
#    --gpu $1 --resume $2 -e --resave
#Acc@1 76.240 Acc@5 93.740

python examples/classifier_cifar100/main.py ~/datasets/data.cifar100 \
    -a cifar100_vggsmall -j 10 -b 128 -p 20 --epochs 200 \
    --gpu $1 --pretrained -e

#Acc@1 76.240 Acc@5 93.740