### 1. Cifar10

1. Baseline BN
```bash
python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
    -a cifar10_vggtiny_bn -j 10 -b 256 -p 20 --epochs 300 \
    --gpu $1 --log-name $2 --lr 0.002 --cosine --debug
```
92.66

2. BatchNorm Fusion
```bash
python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
    -a cifar10_vggtiny_bn -j 10 -b 256 -p 20 --epochs 300 \
    --gpu 0 --resume $1 -e --resave --bn-fusion
```
92.66

3. VGGTiny Test
```bash
python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
    -a cifar10_vggtiny -j 10 -b 256 -p 20 --epochs 300 \
    --gpu 0 --pretrained -e
```
92.66

4. Quantize VGGtiny
```bash
python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
    -a cifar10_vggtiny_llsqs_bwns -j 10 -b 256 --pretrained \
    --gpu $1 --log-name $2 --lr 0.002 --epochs 300 --cosine \
    --qw 1 --qa 4 --q-mode layer_wise
```

5. Extract inner data
```bash
python examples/classifier_cifar10/main.py ~/datasets/data.cifar10 \
    -a cifar10_vggtiny_llsqs_bwns -j 10 -b 32 \
    --gpu 0 --qw 1 --qa 4 --q-mode layer_wise --resume $1 --extract-inner-data -e
```

### 2. Google Speech Command (10 classes)
1. Baseline
```bash
python examples/classifier_speech_command/main.py ~/datasets/data.speech_commands_g_10 \
    -a gcommand_lenet_mfcc -j 1 -b 256 -p 20 --epochs 300 \
    --gpu $1 --log-name $2 --lr 0.002 --cosine --debug
```
91.39

2. Test
```bash
python examples/classifier_speech_command/main.py ~/datasets/data.speech_commands_g_10 \
    -a gcommand_lenet_mfcc -j 1 -b 256 -p 20 --epochs 300 \
    --gpu $1 --resume $2 -e --resave
```
91.39

3. Quantize
```bash
python examples/classifier_speech_command/main.py ~/datasets/data.speech_commands_g_10 \
    -a gcommand_lenet_mfcc_llsqs_bwns -j 1 -b 256 -p 20 --epochs 300 \
    --gpu $1 --log-name $2 --lr 0.002 --cosine --debug \
    --qw 1 --qa 4 --q-mode layer_wise --pretrained
 ```
 85.42

 
 4. Extract inner data
 ```bash
 python examples/classifier_speech_command/main.py ~/datasets/data.speech_commands_g_10 \
    -a gcommand_lenet_mfcc_llsqs_bwns -j 20 -b 32 -p 20 --gpu 0 \
    --qw 1 --qa 4 --q-mode layer_wise --resume $1 --extract-inner-data -e
 ```
