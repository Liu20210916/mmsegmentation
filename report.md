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

# 2. 算法思路、亮点解读、建模算力、环境 （GPU部分）

## 2.1 算法思路

常规的图像分割思路，将图像输入神经网络，输出每个像素各类的预测概率值。

## 2.2 亮点解读

分割性能优化：
采用高斯加权滑窗平均，有效解决大幅遥感图像“切割-预测”流程中切割边缘预测不准确、切割线两侧预测不一致的问题，在预测阶段提升模型分割效果

计算效率优化：
采用并行式图片加载，既支持已切割小图的同步加载，也支持大图的并行切割，在预测阶段有效的提升了模型的数据加载速度，结合批量化预测，进一步充分利用GPU运算能力，在使用相同模型条件下，计算速度显著优于其他选手。

## 2.3 建模算力

- 显存：需要至少32G显存，推荐双卡各24G显存，本地建模使用的是2x TITAN RTX（2x24=48G显存）。

- 耗时：模型训练约8h，预测（单卡）约3min。

## 2.4 环境

```
ubuntu=18.04
python=3.7.10
pytorch=1.7.1
mmcv=1.3.11
mmseg=0.17.0
cuda=10.2
```

# 3. 详细的解题思路说明 （GPU部分）

浏览近期发表在知名会议、期刊上的关于图像分割模型的论文，比较论文中报道的模型性能（例如在CitySpace和ADE20k公开数据集上的效果），参考`mmsegmentation`的代码实现以及对应的结果，选择最佳的模型。

因此选择了Swin-Base+UperNet。分别按照`mmsegmentation`推荐的训练流程训练12个epoch。仅使用默认的训练参数，主要在测试阶段进行性能优化。

# 4. 项目运行环境和运行办法等信息，根据该文档操作指引，能复现选手结果 （GPU部分）

详见各阶段提交的复现代码

# 5. NPU部分

参考[mindspore/models-deeplabv3plus](https://gitee.com/mindspore/models/tree/master/official/cv/deeplabv3plus)，对内部进行少量调整，以适应本次竞赛数据格式，采用默认参数进行训练12个周期。由于对Mindspore框架、ModelArts平台不熟悉，大部分时间花费在了学习使用阶段，因此没有在模型和算法上提出有效的创新。

# 6. 其他说明内容

## 6.1 文档+代码

本文档以及代码在[https://github.com/CarnoZhao/mmsegmentation/tree/remote_review](https://github.com/CarnoZhao/mmsegmentation/tree/remote_review)，**注意branch是remote_review**。


## 6.2 预训练模型

预训练模型链接（可直接下载）：

- Swin-Base：(需要手动下载，见`run.sh`第二行)

    - 论文：[https://arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030)

    - 模型：[https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)



# 7. 联系方式（手机号或微信）

- 手机：18810903806

- 微信号：Carno_Zhao

# 8. 身份证明
