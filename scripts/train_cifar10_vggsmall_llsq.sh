#!/usr/bin/env bash
# 1. Train fp model

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall -j 10 -b 256 \
#    --gpu $1 --log-name $2 --lr 0.0002 --epochs 400
# 93.4

# ./scripts/train_cifar10_vggsmall_llsq.sh 0 baseline
# Checkpoint files are saved in logger/cifar10_vggsmall_baseline

# 2. Validate the fp model

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall -j 10 -b 256 \
#    --gpu $1 --lr 0.0002 --epochs 200 \
#    --pretrained -e
# ./scripts/train_cifar10_vggsmall_llsq.sh 0

# 3. Generate key map

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall_llsq --gen-map --original-model cifar10_vggsmall

# ./scripts/train_cifar10_vggsmall_llsq.sh
# Then we get QuantizedPyTorch/models/weight_keys_map/cifar10_vggsmall_llsq_map.json

# 4. Train quantized vggsmall on cifar10
python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
    -a cifar10_vggsmall_llsq -j 10 -b 256 --pretrained \
    --gpus $1 --log-name $2 --lr 0.0002 --epochs 100 \
    --qw 4 --qa 4 --q-mode kernel_wise
# ./scripts/train_cifar10_vggsmall_llsq.sh 0 try1
# Checkpoint files are saved in logger/cifar10_vggsmall_llsq_w4a4_try1
