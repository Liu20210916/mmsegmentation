# 1. 代码展示
`mmsegmentation`训练流程配置：

```python
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
```

# 2. 算法思路、亮点解读、建模算力、环境

## 2.1 算法思路

常规的图像分割思路，将图像输入神经网络，输出每个像素各类的预测概率值。

## 2.2 亮点解读

亮点：**除模型融合外，没有使用特殊手段就能获得高分，在复赛期间有较大的潜力**

目前的模型训练几乎没有使用非常规的技术，基本按照`mmsegmentation`默认流程。

仅选择了一个CNN模型和一个Transformer模型融合。

## 2.3 建模算力

- 显存：需要至少32G显存，推荐双卡各24G显存，本地建模使用的是2x TITAN RTX（2x24=48G显存）。

- 耗时：两个模型训练（双卡）约8+6=14h，预测（单卡）约20min。

## 2.4 环境

```
ubuntu=18.04
python=3.7.10
pytorch=1.7.1
mmcv=1.3.11
mmseg=0.17.0
cuda=10.2
```

# 3. 详细的解题思路说明

浏览近期发表在知名会议、期刊上的关于图像分割模型的论文，比较论文中报道的模型性能（例如在CitySpace和ADE20k公开数据集上的效果），参考`mmsegmentation`的代码实现以及对应的结果，选择最佳的模型。

因此，从CNN模型中，选择了ResNeSt101+DeepLabV3Plus（记为*res*），从Transformer模型中，选择了Swin-Base+UperNet（记为*swin*）。分别按照`mmsegmentation`推荐的训练流程训练12个epoch。

就单个模型而言，*res*可以获得67.19的分数，*swin*可以获得69.61的分数。对两个模型输出概率进行平均，融合输出得分为70.06。

# 4. 项目运行环境和运行办法等信息，根据该文档操作指引，能复现选手结果


## 4.1 安装环境
```sh
# 安装mm-lab的conda环境
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# 安装pytorch
conda install pytorch=1.7.1 torchvision cudatoolkit=10.2 -c pytorch

# 安装mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/1.7.1/index.html

# 安装mmseg
git clone https://github.com/CarnoZhao/mmsegmentation.git
cd mmsegmentation
git checkout remote_review
pip install -r requirements/build.txt
pip install -v -e .
```

## 4.2 配置数据
```=
|--data
    |--remote
        |--train # (编号7001~34999的图片)
        |   |--images
        |   |  |--xxx.tif
        |   |  |--xxx.tif 
        |   |
        |   |--labels
        |      |--xxx.png
        |      |--xxx.png 
        |
        |--val # (编号0~7000的图片)
        |   |--images
        |   |  |--xxx.tif
        |   |  |--xxx.tif 
        |   |
        |   |--labels
        |      |--xxx.png
        |      |--xxx.png 
        |
        |--test
            |--images
            |  |--xxx.tif
            |  |--xxx.tif 
            |
            |--labels # (测试集本没有标签，为了保证数据加载正确，随意填充10000张png即可)
               |--xxx.png
               |--xxx.png 
```

## 4.3 训练

> 如果要复现训练过程，执行下面的代码

```sh
# 下载swin的预训练模型
mkdir weights
python tools/model_converters/swin2mmseg.py https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth weights/swin_base_patch4_window12_384_22k.pth 

# 训练
gpus=2 # (GPU 数量)
./tools/dist_train.sh ./work_configs/remote/remote_swb.py $gpus
./tools/dist_train.sh ./work_configs/remote/remote_dl3pr101.py $gpus
```

> 如果不复现训练模型，执行下面代码准备预测

```sh
mkdir ./work_dirs/remote/swb384_22k_1x_16bs_all ./work_dirs/remote/dl3pr101_1x_16bs_all
cp ./work_configs/remote/remote_swb.py ./work_dirs/remote/swb384_22k_1x_16bs_all/remote_swb.py
cp ./work_configs/remote/remote_dl3pr101.py ./work_dirs/remote/dl3pr101_1x_16bs_all/remote_dl3pr101.py

# 从百度云下载模型：swin_latest.pth, res_latest.pth
# >>>
# 链接：https://pan.baidu.com/s/1xIgbYNwBUcDPbOkzMImzYQ 
# 提取码：u45x
# >>>
cp swin_latest.pth ./work_dirs/remote/swb384_22k_1x_16bs_all/latest.pth
cp res_latest.pth ./work_dirs/remote/dl3pr101_1x_16bs_all/latest.pth
```

## 4.4 预测

```sh
python ./tools/ensemble_inferencer.py
```

## 4.5 输出

输出结果在`./work_dirs`

# 5. 其他说明内容

## 5.1 文档+代码

本文档以及代码在[https://github.com/CarnoZhao/mmsegmentation/tree/remote_review](https://github.com/CarnoZhao/mmsegmentation/tree/remote_review)，**注意branch是remote_review**。


## 5.2 预训练模型

预训练模型链接（可直接下载）：

- ResNeSt101：(训练过程会自动下载)

    - 论文：[https://arxiv.org/abs/2004.08955](https://arxiv.org/abs/2004.08955)

    - 模型：[https://download.openmmlab.com/mmclassification/v0/resnest/resnest101_imagenet_converted-032caa52.pth](https://download.openmmlab.com/mmclassification/v0/resnest/resnest101_imagenet_converted-032caa52.pth)

- Swin-Base：(需要手动下载，见`run.sh`第二行)

    - 论文：[https://arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030)

    - 模型：[https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)


## **5.3 复现须知**
请完成环境安装后再执行训练和预测，强烈建议环境安装按照[4.1节](#41-安装环境)**一步一步安装**，若遇到问题可参考[get_started.md](https://github.com/CarnoZhao/mmsegmentation/blob/master/docs/get_started.md#installation)。

- 第一步：[安装环境](#41-安装环境)

- 第二步：[配置数据](#42-配置数据)

- 第三步：[训练、预测（执行`run.sh`）](#43-训练)

# 6. 联系方式（手机号或微信）

- 手机：18810903806

- 微信号：Carno_Zhao

# 7. 身份证明
