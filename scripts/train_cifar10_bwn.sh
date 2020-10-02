#!/usr/bin/env bash
#========================
#1. python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggvvtiny -j 10 -b 256 -p 20 --epochs 300 \
#    --gpu $1 --log-name $2 --lr 0.002

#2. python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggvvtiny -j 10 -b 256 -p 20 --epochs 300 \
#    --gpu 0 --resume $1 -e --resave

#3. python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggvtiny -j 10 -b 256 -p 20 --epochs 300 \
#    --gpu 0 --pretrained -e

#4. python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggvvtiny_llsqs_bwns --gen-map --original-model cifar10_vggvvtiny -j 10 -b 256 \
#    --lr 0.002 --epochs 300 \
#    --qw 1 --qa 4 --q-mode layer_wise --floor

#5. python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggvvtiny_llsqs_bwns -j 10 -b 256 --pretrained \
#    --gpu $1 --log-name $2 --lr 0.002 --epochs 300 \
#    --qw 1 --qa 4 --q-mode layer_wise --floor

#6. python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggvvtiny_llsqs_bwns -j 10 -b 32 --pretrained \
#    --gpu $1 --resume $2 --lr 0.002 --epochs 300 \
#    --qw 1 --qa 4 --q-mode layer_wise --floor -e --extract-inner-data





#==============old version=============
# baseline bn
#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggtiny_bn -j 10 -b 256 -p 20 --epochs 300 \
#    --gpu $1 --log-name $2 --lr 0.002 --cosine --debug
# 92.66

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggtiny_bn -j 10 -b 256 -p 20 --epochs 300 \
#    --gpu 0 --resume $1 -e --resave --bn-fusion
# 92.66

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggtiny -j 10 -b 256 -p 20 --epochs 300 \
#    --gpu 0 --pretrained -e
# 92.66

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggtiny_llsqs_bwns -j 10 -b 256 --pretrained \
#    --gpu $1 --log-name $2 --lr 0.002 --epochs 300 \
#    --qw 1 --qa 4 --q-mode layer_wise --floor

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggtiny_llsqs_bwns -j 10 -b 32 \
#    --gpu 0 --qw 1 --qa 4 --q-mode layer_wise --floor \
#    --resume $1 -e --extract-inner-data