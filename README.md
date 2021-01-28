# Dense Regression Network for Video Grounding

This repo holds the codes and models for the DRN framework presented on CVPR 2020

**Dense Regression Network for Video Grounding**
Runhao Zeng, Haoming Xu, Wenbing Huang, Peihao Chen, Mingkui Tan, Chuang Gan, *CVPR 2020*, Seattle, Washington.

[[Paper](https://arxiv.org/abs/2004.03545)]

[//]: ------------------------------Separator------------------------------

# Contents

---
- [Usage Guide](#usage-guide)
    - [Module Preparation](#module-preparation)
    - [Code Preparation](#code-preparation)
    - [Training DRN](#training-drn)
        - [First Stage](#first-stage)
        - [Second Stage](#second-stage)
        - [Third Stage](#third-stage)
    - [Testing DRN](#testing-drn)
- [Other Info](#other-info)
    - [Citation](#citation)
    - [Contact](#contact)
---

[//]: ------------------------------Separator------------------------------

# Usage Guide

## Code and Data Preparation

[[back to top](#dense-regression-network-for-video-grounding)]

### Get the code

Clone this repo with git

```bash
git clone https://github.com/Alvin-Zeng/DRN
cd DRN
```

### Download Features

Here, we provide the C3D features on Charades-STA for training and testing.

Charades-STA: You can download it from [Baidu Cloud][features_baidu] (password: smil).

## Module Preparation

[[back to top](#dense-regression-network-for-video-grounding)]


Start from a clear conda env

```bash
conda create -n DRN
conda activate DRN
```

This repo is based on FCOS, use the following command to install it

```bash
bash setup.sh
```

Other minor Python modules can be installed by running

```bash
pip install -r requirements.txt
```


## Training DRN

[[back to top](#dense-regression-network-for-video-grounding)]

Plesse first set the path of features in data/default_config.yaml

```bash
feature_root: $PATH_OF_FEATURES
```
### First Stage

Use the following command to train the first stage of DRN

```bash
bash drn_train.sh $PATH_TO_SAVE_FIRST_MODEL is_first_stage
```

- `$PATH_TO_SAVE_FIRST_MODEL` denotes the path to save the first-stage model


### Second Stage

Use the following command to train the second stage of DRN

```bash
bash drn_train.sh $PATH_TO_SAVE_SECOND_MODEL is_second_stage $FIRST_CHECKPOINT 
```

- `$PATH_TO_SAVE_SECOND_MODEL` denotes the path to save the second-stage model

- `$FIRST_CHECKPOINT` denotes the trained model from the first stage


### Third Stage

Use the following command to train the third stage of DRN

```bash
bash drn_train.sh $PATH_TO_SAVE_THIRD_MODEL is_third_stage $SECOND_CHECKPOINT
```

- `$PATH_TO_SAVE_THIRD_MODEL` denotes the path to save the third-stage model

- `$SECOND_CHECKPOINT` denotes the trained model from the second stage


## Testing DRN

[[back to top](#dense-regression-network-for-video-grounding)]

Here, we provide the models trained on Charades-STA for testing.

Charades-STA: You can download them from [Baidu Cloud][models_baidu] (password: smil).


Use the following command to test the trained model

```bash
bash drn_test.sh $TRAINED_CHECKPOINT
```

- `$TRAINED_CHECKPOINT` denotes the trained model

The evaluation results will be put in the "results" folder

### Charades-STA

| Method        | R@1 IoU=0.5 (%) | R@5 IoU=0.5 (%) |
|:-------------:|:---------------:|:---------------:|
| DRN (C3D)     |      45.40      |      89.06      |

[//]: ------------------------------Separator------------------------------

# Other Info

[[back to top](#dense-regression-network-for-video-grounding)]

## Citation

Please cite the following paper if you feel DRN useful to your research

```
@inproceedings{DRN2020CVPR,
  author    = {Runhao Zeng and
               Haoming Xu and
               Wenbing Huang and
               Peihao Chen and
               Mingkui Tan and
               Chuang Gan},
  title     = {Dense Regression Network for Video Grounding},
  booktitle = {CVPR},
  year      = {2020},
}
```


## Contact

For any question, please file an issue or contact

```
Runhao Zeng: runhaozeng.cs@gmail.com
```

[features_baidu]: https://pan.baidu.com/s/1Sn0GYpJmiHa27m9CAN12qw
[models_baidu]: https://pan.baidu.com/s/1EQNi5cLEDptVed91YpHVwg
