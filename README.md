# Cheaper Clicks from Boxes: Cyclic Querying for Interactive Small Object Detection

## Introduction
This is the official implementation of our paper titled *Cheaper Clicks from Boxes: Cyclic Querying for Interactive Small Object Detection*, dubbed as ***CLIQ***.

## Dependencies
 - CUDA 10.2
 - Python 3.8
 - PyTorch 1.10.0
 - TorchVision 0.11.0
 - mmrotate 0.3.0
 - mmcv-full 1.5.0
 - numpy 1.22.4

## Installation
This repository is build on **MMRotate 0.3.0**  which can be installed by running the following scripts. Please ensure that all dependencies have been satisfied before setting up the environment.
```
git clone https://github.com/shaunyuan22/CLIQ
cd CLIQ
pip install -v -e .
```
Please refer to [mmrotate](https://github.com/open-mmlab/mmrotate) for more details about MMRotate installation.

## Training and Evaluation
 - Training:
```
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
```

 - Evaluation:
```
bash ./tools/dist_test_noc.sh ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} --eval mAP
```

## Acknowledgement
The evaluation scripts for interaction framework are partially forked from [C3Det](https://github.com/ChungYi347/Interactive-Multi-Class-Tiny-Object-Detection), and we appreciate their helpful and open-sourced work.
