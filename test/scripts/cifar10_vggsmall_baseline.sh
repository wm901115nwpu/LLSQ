#!/usr/bin/env bash
# In LQ-Net Learning rate is configured as follows:
# [(1, 0.02), (80, 0.002), (160, 0.0002), (300, 0.00002)]) max epoch: 400
# 0.002 ==warm-up==> 0.02 ==cosine==> 0 don't work
# 0.02 ==warm-up==> [(1, 0.2), (80, 0.02), (160, 0.002), (300, 0.0002)]

echo 'cifar10_vggsmall_baseline on single GPU'
read -r -p "Are You Sure? [Y/n] " input
case $input in
    [yY][eE][sS]|[yY])
		echo "Yes"
		python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
        -a cifar10_vggsmall -j 10 -b 100 -p 20 --epochs 400 \
        --gpu 0 --log-name 'sing_bl_test' --lr 0.02 --lr-scheduler 'MultiStepLR' --milestones 80 160 300 --debug \
        --warmup-epoch 10
        # 93.14
		;;
    [nN][oO]|[nN])
		echo "No"
       	;;
    *)
		echo "Invalid input..."
		exit 1
		;;
esac



echo 'cifar10_vggsmall_baseline on multi GPU'
read -r -p "Are You Sure? [Y/n] " input
case $input in
    [yY][eE][sS]|[yY])
		echo "Yes"
		python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
        --dist-url 'tcp://127.0.0.1:9081' --dist-backend 'nccl' \
        --multiprocessing-distributed --world-size 1 --rank 0 \
        -a cifar10_vggsmall -j 10 -b 100 -p 20 --epochs 400 \
        --log-name 'multi_bl_test' --lr 0.02 --lr-scheduler 'MultiStepLR'  --milestones 80 160 300 --debug \
        --warmup-epoch 10
        # (93.40 ± 0.1)
		;;
    [nN][oO]|[nN])
		echo "No"
       	;;
    *)
		echo "Invalid input..."
		exit 1
		;;
esac

