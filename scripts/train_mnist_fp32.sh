#!/usr/bin/env bash
#python examples/classifier_mnist/main.py ~/datasets/data.mnist \
#    -a mnist_resnet18 -j 10 -b 128 -p 20 --epochs 100 \
#    --gpu $1 --log-name $2 --lr 0.002 --cosine

#python examples/classifier_mnist/main.py ~/datasets/data.mnist \
#    -a mnist_resnet18_tbq -j 10 -b 128 -p 20 --epochs 100 \
#    --gpu $1 --log-name $2 --lr 0.002

#python examples/classifier_mnist/main.py ~/datasets/data.mnist \
#    -a mnist_lenet -j 10 -b 128 -p 20 --epochs 100 \
#    --gpu $1 --log-name $2 --lr 0.002

python examples/classifier_mnist/main.py ~/datasets/data.mnist \
    -a mnist_lenet_tbq -j 10 -b 128 -p 20 --epochs 100 \
    --gpu $1 --log-name $2 --lr 0.002