# SQ: Sparsity(INS) & Quantization(LSQ) 

- **ResNet18**

| S\Q  | w32a32 | w6a6  | w4a4  | w3a3  |
|------|--------|-------|-------|-------|
| 0    | 69.758 | 70.13 | 70.17 | 69.4  |
| 0.3  | 70.3   | 69.98 |       |       |
| 0.4  | 70.13  | 69.8  | 70.17 |       |
| 0.5  | 70.26  | 70.31 |       |       |
| 0.6  | 70.57  | 69.9  |       |       |
| 0.65 | 70.41  |       | 69.72 | 68.68 |
| 0.7  | 70.21  |       | 69.49 | 68.44 |
| 0.75 | 69.8   |       | 69.19 | 68.15 |
| 0.85 | 68.54  |       |       |       |
| 0.90 | 66.64  |       |       |       |
| 0.95 | 62.25  |       |       |       |

- **ResNet50**

| S\Q  | w32a32 |
|------|--------|
| 0    | 76.13  |
| 0.85 | 75.33  |
| 0.90 | 74.42  |
| 0.95 | 71.81  |

- **EfficientNet**

| S\Q  | w32a32 |
|------|--------|
|  0   | 77.684 | 
|  0.6 | 74.45  |

## INS: Incrental Network (Sparsity) pruning

- Lazy (dynamic & incremental) mask

<img src="https://latex.codecogs.com/svg.latex?\fn_phv&space;S_{iter}&space;=&space;(S_{exp}&space;-&space;S_{init})&space;\sin&space;(\frac{\pi}{2}\cdot\frac{iter}{\beta\cdot&space;iter_{total}})&space;&plus;&space;S_{init}" title="S_{iter} = (S_{exp} - S_{init}) \sin (\frac{\pi}{2}\cdot\frac{iter}{\beta\cdot iter_{total}}) + S_{init}" />

- INS is **not** better than level (percent) prunning in all cases, they perform consistently sometimes.

```bash
python examples/classifier_imagenet/main_sq.py ~/datasets/data.imagenet \
    -a resnet18 -j 10 -b 512 --pretrained \
    --lr 0.001 --wd 1e-4 --warmup-epoch -1 \
    --gpu $1 --log-name $2 --epochs 90 --lr-scheduler CosineAnnealingLR \
    --qw 3 --qa 3 --sparsity 0.3 \
    --debug
```
