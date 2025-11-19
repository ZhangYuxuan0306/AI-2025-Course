# 人工智能课程 第四组-融合深度信息的图像分割

## 🧠 项目简介
本项目基于DeepLabv3实现了DeepCrack裂缝数据集的语义分割，以及基于Mask2Former实现了电子元件数据集的实例分割；在此基础上融合多种深度信息增强方法，有效提升了模型的分割精度，完成任务书的既定要求。

## ⚙️ 环境与数据准备
### Requirements
Python 3.7+, CUDA 9.2+, Pytorch 1.8+ \
Our implementation is based on the MMDetection and MMSegmentation.
For more detailed information, please refer to [MMDet](https://mmdetection.readthedocs.io/en/latest/get_started.html) and [MMSeg](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

### 基础 Conda Env 
```
conda creat --name MMLab python=3.8 -y
conda activate MMLab
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmengine
mim install mmcv
```
### 实例分割任务（基于MMDet）
#### 客制化使用
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```
1. 将文件夹/custom全部放入/mmdetection;
2. 将Instance_seg/configs中的配置放入/mmdetection/configs中;
3. 适配地址
#### 数据集
[电子元器件数据集](https://bhpan.buaa.edu.cn/link/AA130E079FE9264651B535123EDA2D7790)

### 语义分割任务（基于MMSeg）
#### 客制化使用
```
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```
1. 将Semantic_seg/configs中的配置放入/mmsegmentation/configs中;
2. 将Semantic_seg/mmseg/datasets中的数据集处理文件放入/mmsegmentation/mmseg/datasets中，并修改/datasets/__ init __.py：
```
from .DeepCrack import DeepCrackDataset

__all__ = [
    ......, 'DeepCrackDataset'
]
```
3. 适配地址

#### 数据集
[DeepCrack-裂缝数据集](https://bhpan.buaa.edu.cn/link/AA130E079FE9264651B535123EDA2D7790)

## 🛠️ 训练&测试Command
### 训练命令
**Single GPU**
```
python tools/train.py configs/[所需配置文件].py
```
**Multiple GPU**

```
./tools/dist_train.sh configs/[所需配置文件].py GPU_NUM
```

### 测试方法
```
python tools/test.py configs/[所需配置文件].py [model path] --out [NAME].pkl
```

### 权重获取
| Task | Methods | mAP | Weight |
| ---- | ------- | ------ |----- |
| Instance_seg | Baseline | 30.9 | [Link](https://drive.google.com/file/d/14MfED97WzMAyFtSdjSAzqSL6EiL7r2CA/view?usp=drive_link) |
|       | Channel Merge | 41.6 | [Link](https://drive.google.com/file/d/1c3noDtdTyb-zmVh7GcC2QouVK5igKFUa/view?usp=sharing) |
|       | Feature Fusion | 44.4 | [Link](https://drive.google.com/file/d/1Neud97zA-RkWEJV0-nGh0dbOk_8nlYuc/view?usp=sharing) |
|       | Geometric Guidance | 43.3 | [Link](https://drive.google.com/file/d/1jEIXl_d5LZN4ZrssrZG7TSVu7P2r3Cc1/view?usp=drive_link) |
| Semantic_seg | with_load.pth | 76.4 | [Link](https://drive.google.com/file/d/1wTZvPPpYpnCGKBxHZZpvNL50yhhyo1Sx/view?usp=sharing) |
|              | wo_load.pth | 76.1 | [Link](https://drive.google.com/file/d/11XUBG2XhuC36P8UyBSJFWC6Al6--5f3A/view?usp=drive_link) |

## 📊 效果展示
### 实例分割
<p align="center">
<img src=./Instance_seg/image.jpeg>
</p>

### 语义分割
<p align="center">
<img src=./Semantic_seg/image2.jpeg>
</p>

### 前端
<p align="center">
<img src=./image3.jpeg>
</p>

## ❤️ 致谢
* MMdetection: [mmdetection](https://mmdetection.readthedocs.io/en/latest/)
* MMsegmentation: [mmsegmentation](https://mmsegmentation.readthedocs.io/en/latest/)
* Mask2Former: [Code](https://github.com/facebookresearch/Mask2Former); [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf)
* DFormerv2: [Code](https://github.com/VCIP-RGBD/DFormer); [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_DFormerv2_Geometry_Self-Attention_for_RGBD_Semantic_Segmentation_CVPR_2025_paper.pdf)