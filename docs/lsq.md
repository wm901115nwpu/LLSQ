# LSQ
| **LSQ**  | fp32         | w4a4 | w3a3 | w2a2 | w8a8(1epoch, quantize data) |
|----------|--------------|------|------|------|--------------|
| AlexNet  | 56.55, 79.09 | **56.96**, 79.46 [√](https://tensorboard.dev/experiment/MNSkwpg9SJySk201OqJLhw/) | 55.31, 78.59 |  51.18, 75.38 | |
| ResNet18 | 69.76, 89.08 | **70.26**, 89.34 [√](https://tensorboard.dev/experiment/bRQtjRFDRHGbJyQ6Jd3ztQ/)   |69.45, 88.85 |      | 69.68  88.92 [√](https://tensorboard.dev/experiment/jqrFL5q1QwSZRz3wSW6LQw/) |
|          |              |      |      |      |    |
|          |              |      |      |      |    |


## bash

### AlexNet_LSQ_w4a4
```bash
python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
    -a alexnet_lsq -j 10 --pretrained -b 2048 --log-name $2 \
    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
    --gpu $1 --epochs 90 --lr-scheduler CosineAnnealingLR \
    --qw 4 --qa 4 --q-mode layer_wise \
    --debug
```

### AlexNet_LSQ_w3a3
```bash
python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
    -a alexnet_lsq -j 10 --pretrained -b 2048 --log-name $2 \
    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
    --gpu $1 --epochs 90 --lr-scheduler CosineAnnealingLR \
    --qw 3 --qa 3 --q-mode layer_wise \
    --debug
```

### AlexNet_LSQ_w2a2
```bash
python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
    -a alexnet_lsq -j 10 --pretrained -b 2048 --log-name $2 \
    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
    --gpu $1 --epochs 90 --lr-scheduler CosineAnnealingLR \
    --qw 3 --qa 3 --q-mode layer_wise \
    --debug
```

### ResNet18_LSQ_w4a4
```bash
python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
    -a resnet18_lsq -j 10 -b 512 --pretrained \
    --lr 0.01 --wd 1e-4 --warmup-epoch -1 \
    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
    --qw 4 --qa 4 --q-mode layer_wise \
    --debug
```
