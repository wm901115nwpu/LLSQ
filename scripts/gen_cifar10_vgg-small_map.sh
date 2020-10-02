#!/usr/bin/env bash
#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall_q --gen-map --original-model cifar10_vggsmall
#
#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall_qfn --gen-map --original-model cifar10_vggsmall

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall_qfnv2 --gen-map --original-model cifar10_vggsmall

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall_qfi --gen-map --original-model cifar10_vggsmall


#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall_qfn_pact --gen-map --original-model cifar10_vggsmall

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall_qv2 --gen-map --original-model cifar10_vggsmall

# For Bit pruning

#python examples/classifier_cifar10/bp_main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall_bp --gen-map --original-model cifar10_vggsmall

#
#python examples/classifier_cifar10/bp_main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggsmall_bp_exp --gen-map --original-model cifar10_vggsmall

#python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
#    -a cifar10_vggtiny_llsqs_bwns --gen-map --original-model cifar10_vggtiny

python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
    -a cifar10_vggsmall_llsq --gen-map --original-model cifar10_vggsmall