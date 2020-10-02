## install
```bash
git clone https://github.com/hustzxd/QuantizedPyTorch.git

cd QuantizedPyTorch
pip install -r requirements.txt
```

## setup
```bash
source setup.sh 
```



## Prepare pre-trained model
If you don't want train cifar10 model from scratch, you can download from [pan.baidu](https://pan.baidu.com/s/1o79DEFS)

| model-name                 | accuracy |
|----------------------------|----------|
| cifar10_vggsmall-zxd-fiefefe.pth  | 93.43    |

Please move these models to `.torch/model/` or `.cache/torch/checkpoint/`.

## Environment of demo
```bash
Collecting environment information...
PyTorch version: 1.4.0
Is debug build: No
CUDA used to build PyTorch: 10.1

OS: Ubuntu 16.04.5 LTS
GCC version: (Ubuntu 5.4.0-6ubuntu1~16.04.12) 5.4.0 20160609
CMake version: version 3.5.1

Python version: 3.6
Is CUDA available: Yes
CUDA runtime version: Could not collect
GPU models and configuration: 
GPU 0: PH402 SKU 200
GPU 1: PH402 SKU 200
GPU 2: PH402 SKU 200
GPU 3: PH402 SKU 200
GPU 4: PH402 SKU 200
GPU 5: PH402 SKU 200
GPU 6: PH402 SKU 200
GPU 7: PH402 SKU 200
GPU 8: PH402 SKU 200
GPU 9: PH402 SKU 200
GPU 10: PH402 SKU 200
GPU 11: PH402 SKU 200
GPU 12: PH402 SKU 200
GPU 13: PH402 SKU 200
GPU 14: PH402 SKU 200
GPU 15: PH402 SKU 200
GPU 16: PH402 SKU 200
GPU 17: PH402 SKU 200
GPU 18: PH402 SKU 200
GPU 19: PH402 SKU 200

Nvidia driver version: 430.09
cuDNN version: Could not collect

Versions of relevant libraries:
[pip] numpy==1.18.5
[pip] torch==1.4.0
[pip] torchsummary==1.5.1
[pip] torchvision==0.5.0
```
