#!/usr/bin/env bash

echo 'cifar10_vggsmall_npu_structured_pruner NPU 7 on single GPU  ADMM drho0.05'
read -r -p "Are You Sure? [Y/n] " input
case $input in
    [yY][eE][sS]|[yY])
		echo "Yes"
		# admm step 1
        python examples/classifier_cifar10/main_admm_npu.py ~/datasets/data.cifar10 \
        -a cifar10_vggsmall -j 10 -b 128 --pretrained \
        --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
        --gpu 0 --log-name 'npu_7_admm_test' --epochs 90 --lr-scheduler CosineAnnealingLR \
        --debug --admm --non-zero-num 7 --rho 0.05 --rho-dynamic
        # 94.22
		;;
    [nN][oO]|[nN])
		echo "No"
       	;;
    *)
		echo "Invalid input..."
		exit 1
		;;
esac

