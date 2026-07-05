# Attack and evaluation code

Original 2017 implementation (TensorFlow 1.x + cleverhans), released as a historical artifact. The scripts expect the checkpoints and dataset in this directory.

## Setup

1. Download `ckpts.zip` and `dataset.zip` from the [v1.0 Release](https://github.com/thenamangoyal/adversarial-attack/releases/tag/v1.0) and extract them here, so you have:

```
code/
  ckpts/      inception_v3.ckpt, inception_v2.ckpt, resnet_v2_152.ckpt
  dataset/    images/ (originals), modified_images/{FGM,BIM,VAM,BinIM}/
```

2. Dependencies (era-appropriate): TensorFlow 1.x and cleverhans
   (`pip install -e git+http://github.com/tensorflow/cleverhans.git#egg=cleverhans`).

## Run an attack

```
./run_attack_fgm.sh     # Fast Gradient Method
./run_attack_bim.sh     # Basic Iterative Method
./run_attack_vam.sh     # Virtual Adversarial Method
./run_attack_binim.sh   # Binary Iterative Method (ours)
```

## Evaluate modified images

```
./evaluate_InceptionV3.sh XYZ
./evaluate_InceptionV2.sh XYZ
./evaluate_Resnet.sh XYZ
```

Replace `XYZ` with the method whose outputs you want to evaluate: `FGM`, `BIM`, `VAM`, or `BinIM`. Example: `./evaluate_InceptionV3.sh BinIM`.
