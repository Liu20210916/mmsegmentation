# 1. 代码展示
`mmsegmentation`训练流程配置：

```python
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

# 2. GPU部分技术文档

## 2.1 算法思路

在本次竞赛中，我队使用深度神经网络中的编码器-解码器（Encoder-Decoder）结构进行遥感图像的像素级语义分割。在模型中，图像首先经过Encoder多次下采样，被编码为高层次的语义信息，再在Decoder中结合低层次的语义信息，实现不同层次特征的交互，经过多次上采样输出各个像素分属于47个类别的概率，得到最终的输出结果。

在模型方面，我们的模型中Encoder部分选择了Swin-Transformer，因为它是目前开源的网络结构中特征提取效果最好的，我们的Decoder部分选择了UperNet结构，因为在Swin-Transformer论文中，作者结合Swin-Transformer和UperNet在开源语义分割数据集ADE20K上达到了最好的效果，所以我们依据论文，同样使用了Swin-Transformer和UperNet的组合。

在数据方面，我们采用了常规的几何变换、颜色变换等数据增强策略。特别的，因为本次竞赛提供的数据为4通道的遥感图像，不同于一般的RGB三通道自然图像，所以并不能直接使用由ImageNet预训练的模型。再考虑到神经网络浅层部分是否能有效的提取出图像特征对整个网络的训练至关重要，所以，为了将预训练模型中的三通道权重转换为四通道，我们并没有采用随机初始化策略，而是将**红色通道的权重迁移到远红外通道**上。这样训练得到的模型有着更高的分割效果。

在预测推理方面，由于竞赛中既要求模型的分割准确率，也要求模型的运算速度，所以我们在预测推理方面做了尽可能多的优化。

一方面，因为遥感图像本身以较大尺寸储存，不能适应神经网络的硬件限制，所以常规的分割方案会将大图切割为小图再分别进行预测，这样会使得各个小图会损失大量的边界信息，导致边缘部分预测不准确，而且会使得切割线两侧的小图预测不一致，出现明显的人工切割分界线。为了解决这样的问题，我们提出了**高斯加权的滑窗平均策略**，对一张大图，采用带重叠的滑动窗格切割小图，保证大图中的所有位置至少存在于一张小图的中心，并且，对每个小图的预测结果，我们使用高斯分布生成二维权重，加强小图中心的预测置信度，减弱小图边缘的预测置信度，避免了人工引入的边界效应。根据实验，我们的高斯加权滑窗策略可以有效的提升模型的分割效果。

另一方面，一般而言，整个推理过程各图片串行进行，先读取图片再执行推理计算，这样极大的浪费了运算资源，使得GPU的使用率较低。所以我们优化了整个预测流程，采用**并行读取图片策略**并完成小图切割，将切割后的图像保存至内存，将读取过程加快了近一倍。而后，我们再采用批处理的方式调用GPU执行深度学习计算，因为数据已经保存至内存中，并且批处理可以更充分利用GPU的计算优势，所以计算速度相比于单张图片依次处理更快速，计算资源使用率也更高。相比于使用相同模型的其他选手，我们的计算速度优化使得我们的预测速度大幅领先近20%，并且这样的优化策略并不影响分割准确率，使得我们既保证了分割效果，又保证了计算速度。

## 2.2 亮点解读（摘自算法思路）

### **亮点1** 四通道预训练适应

因为本次竞赛提供的数据为4通道的遥感图像，不同于一般的RGB三通道自然图像，所以并不能直接使用由ImageNet预训练的模型。再考虑到神经网络浅层部分是否能有效的提取出图像特征对整个网络的训练至关重要，所以，为了将预训练模型中的三通道权重转换为四通道，我们并没有采用随机初始化策略，而是将红色通道的权重迁移到远红外通道上。这样训练得到的模型有着更高的分割效果。

我们在实验中比较了红通道权重迁移、绿通道权重迁移、蓝通道权重迁移、红绿蓝平均迁移、随机初始化四种四通道策略，发现红色通道权重迁移可以达到更好的模型效果。

### **亮点2** 高斯加权的滑窗平均策略

因为遥感图像本身以较大尺寸储存，不能适应神经网络的硬件限制，所以常规的分割方案会将大图切割为小图再分别进行预测，这样会使得各个小图会损失大量的边界信息，导致边缘部分预测不准确，而且会使得切割线两侧的小图预测不一致，出现明显的人工切割分界线。为了解决这样的问题，我们提出了高斯加权的滑窗平均策略，对一张大图，采用带重叠的滑动窗格切割小图，保证大图中的所有位置至少存在于一张小图的中心，并且，对每个小图的预测结果，我们使用高斯分布生成二维权重，加强小图中心的预测置信度，减弱小图边缘的预测置信度，避免了人工引入的边界效应。根据实验，我们的高斯加权滑窗策略可以有效的提升模型的分割效果。

因为本次比赛中测试集以切割后的小图像为主，所以并不能充分展现我们的高斯加权策略的优势。在实际应用中，我们的方法应该会有更好的效果。

### **亮点3** 计算性能优化
一般而言，整个推理过程各图片串行进行，先读取图片再执行推理计算，这样极大的浪费了运算资源，使得GPU的使用率较低。所以我们优化了整个预测流程，采用并行的策略读取图片并完成小图切割，将切割后的图像保存至内存，将读取过程加快了近一倍。而后，我们再采用批处理的方式调用GPU执行深度学习计算，因为数据已经保存至内存中，并且批处理可以更充分利用GPU的计算优势，所以计算速度相比于单张图片依次处理更快速，计算资源使用率也更高。相比于使用相同模型的其他选手，我们的计算速度优化使得我们的预测速度大幅领先近20%，并且这样的优化策略并不影响分割准确率，使得我们既保证了分割效果，又保证了计算速度。

## 2.3 建模算力

- 显存：需要至少32G显存，本地建模使用的是2x TITAN RTX（2x24=48G显存）。

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

## 2.5 复现教程

```sh
# 下载swin的预训练模型
mkdir weights
python tools/model_converters/swin2mmseg.py https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth weights/swin_base_patch4_window12_384_22k.pth 
# 训练
gpus=2 # (GPU 数量)
./tools/dist_train.sh ./work_configs/remote/remote2_swb.py $gpus

# 预测
python ./tools/run.py
```

# 3. NPU部分技术文档

## 3.1 算法思路

> 注：因为本次竞赛使用Mindspore和ModelArts较为仓促，前期没有积累足够多的使用经验，在NPU上不能完全复现GPU上的分割效果。

在使用Mindspore和Ascend的过程中，我们尝试完全实现GPU中的算法方案。结果如下：

经过实验，使用Mindspore框架，复现Swin-Transformer UperNet之后，本地的硬件设备上可以得到较好的效果，但是难以迁移到Ascend硬件上。我们推测原因如下：部分算子在GPU上执行性能较好，但是在Ascend上计算较慢，尤其涉及到矩阵非数值运算的算子，例如转置(Transpose)，形变(Reshape)等。这些算子不参与矩阵数值的改变，而只影响维度、大小等，按照Numpy的设计，这些信息的变化实际上只是设置了在内存空间中矩阵元素的读取顺序、步长等，无需过多的调整。但是可能是优于MindSpore在Ascend上优化不够完善，使得这些算子执行了过量的计算。

因为这样的问题，导致大量涉及Attention结构的网络无法使用，例如Transformer，OCRNet等优秀的分割网络。所以我们不得已在最后选择了开源的DeeplabV3+模型，经过默认参数的训练，达到了0.12左右的fIoU。

## 3.2 算力和环境

ModelArts单卡Ascend910环境。

## 3.3 复现教程

```sh
# 构建数据来源表
train_image_path="./train/images"
train_label_path="./train/labels"
paste <(ls ${train_image_path} | awk '{print "images/"$0 }' | sort) <(ls ${train_label_path} | awk '{print "labels/"$0 }' | sort) -d " " > 
data_list.txt

# 将数据转换为MindRecords
/home/ma-user/miniconda3/envs/Mindspore-python3.7-aarch64/bin/python ./src/tools/get_dataset_mindrecord.py --data_root "./train" --data_lst "./data_list.txt" --dst_path "./train.msrec" --shuffle False

# 训练
/home/ma-user/miniconda3/envs/Mindspore-python3.7-aarch64/bin/python ./train.py \
    --train_dir ./work_dirs \
    --data_file ./remote2/train.msrec \
    --batch_size 16 \
    --image_mean 123.675 116.28 103.53 114.50 \
    --image_std 58.395 57.12 57.375 57.63 \
    --num_classes 47 \
    --train_epochs 12 \
    --base_lr 1e-2 \
    --ckpt_pre_trained ./weights/resnet101_ascend_v120_imagenet2012_official_cv_bs32_acc78.ckpt

# 测试
cd ./save_dir
mv $(ls -rt | tail -n 1) DeepLabV3plus_s16_2-12_2187.ckpt
cd ..
bash inference.sh
```

# 6. 其他说明内容

## 6.1 文档+代码

本文档以及GPU代码已经公开在[https://github.com/CarnoZhao/mmsegmentation/tree/remote_review](https://github.com/CarnoZhao/mmsegmentation/tree/remote_review)，**注意branch是remote_review**。


## 6.2 预训练模型

预训练模型链接（可直接下载）：

- Swin-Base：(需要手动下载，见`run.sh`第二行)

    - 论文：[https://arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030)

    - 模型：[https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)



# 7. 联系方式（手机号或微信）

- 手机：18810903806

- 微信号：Carno_Zhao
