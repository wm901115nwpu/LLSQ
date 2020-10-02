#!/usr/bin/env bash
python examples/classifier_flower/main.py ~/datasets/data.flowers \
    -a flower102_vggsmall -j 10 -b 128 -p 20 --epochs 200 \
    --gpu $1 --log-name $2 --lr 0.01 --wd 5e-4 \
    --warmup-epoch 10

#python examples/classifier_flower/main.py ~/datasets/data.flowers \
#    -a flower102_vggsmall -j 10 -b 32 -p 20 -e --resume $2 \
#    --gpu $1
# Acc@1 82.295 Acc@5 94.383

#python examples/classifier_flower/main.py ~/datasets/data.flowers \
#    -a flower102_vggsmall -j 10 -b 32 -p 20 -e --pretrained \
#    --gpu $1
# Acc@1 85.836 Acc@5 95.482
