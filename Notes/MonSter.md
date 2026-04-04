# MonSter 代码详解

MonSter 将单目深度估计与立体匹配相结合，实现零样本泛化的立体匹配。

---

## 文件结构总览

```
MonSter/
├── core/
│   ├── submodule.py          # 基础卷积构件：BasicConv, Conv2x, groupwise_correlation
│   ├── utils/
│   │   ├── utils.py          # 工具函数：InputPadder, bilinear_sampler, coords_grid
│   │   └── file_io.py        # 文件读写：pklload, decompress, load_pkl
│   ├── warp.py               # 视差扭曲：disp_warp, meshgrid, normalize_coords
│   ├── extractor.py          # 特征提取：ResidualBlock, BasicEncoder
│   ├── update.py              # 迭代更新：ConvGRU, SepConvGRU, BasicMultiUpdateBlock
│   ├── geometry.py            # 几何编码体：Combined_Geo_Encoding_Volume
│   ├── refinement.py          # 深度图精化：REMP, Attention_HourglassModel
│   ├── monster.py             # 主模型：Monster, compute_scale_shift, hourglass
│   └── stereo_datasets.py    # 数据集加载
├── Depth-Anything-V2-list3/
│   └── depth_anything_v2/
│       ├── dpt.py             # 单目深度解码器
│       └── dinov2.py          # ViT 主干网络
├── train_kitti.py             # 训练脚本
└── evaluate_stereo.py         # 评估脚本
```

---

## 第一阶段：基础工具与子模块

---

### 1. core/submodule.py

#### 1.1 BasicConv（第 7-30 行）

**代码：**
```python
class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)
        return x
```

**输入：**
- `x`：[B, C_in, H, W] 或 [B, C_in, D, H, W]（2D 或 3D）

**输出：**
- `x`：[B, C_out, H, W] 或 [B, C_out, D, H, W]

**数据变换过程：**
```
输入张量
    ↓
卷积层 (Conv2d/Conv3d 或 ConvTranspose2d/ConvTranspose3d)
    ↓
[可选] BatchNorm2d/BatchNorm3d
    ↓
[可选] LeakyReLU (alpha=0.01)
    ↓
输出张量
```

**设计要点：**
- `bias=False`：因为紧接着是 BatchNorm，偏置是冗余的
- `LeakyReLU`：避免神经元坏死，负半轴斜率 0.01

---

#### 1.2 Conv2x（第 33-67 行）

**代码：**
```python
class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x
```

**输入：**
- `x`：[B, C_in, H, W]（需要下采样的特征）
- `rem`：[B, C_rem, H', W']（跳跃连接的特征）

**输出：**
- `x`：[B, C_out, H', W']

**数据变换过程：**
```
x ──────────────────────────────────────────┐
    │                                       │
    ▼                                       │
┌─────────────┐                             │
│  conv1      │  # 下采样 (stride=2)         │
│  BasicConv  │                             │
└─────────────┘                             │
    │                                       │
    ▼                                       │
┌─────────────┐                             │
│ 形状对齐    │  # 如需调整尺寸              │
│ F.interpolate│                            │
└─────────────┘                             │
    │                                       │
    ├───────────────────────┐               │
    ▼                       ▼               │
┌───────────┐      ┌───────────┐            │
│ torch.cat │      │ x + rem   │  # 根据模式 │
│ (concat)  │      │ (add)     │            │
└───────────┘      └───────────┘            │
    │                       │               │
    └───────────┬───────────┘               │
                ▼                           │
        ┌─────────────┐                      │
        │  conv2      │  # 1x1 卷积整合     │
        │  BasicConv  │                      │
        └─────────────┘                      │
                │                            │
                └────────────────────────────┘
```

**参数说明：**
| 参数 | 说明 |
|------|------|
| `concat=True` | 拼接后通道数 ×2 |
| `concat=False` | 相加后通道数不变 |
| `keep_dispc=True` | 保持视差通道的特殊处理 |

---

#### 1.3 groupwise_correlation（第 150-156 行）

**代码：**
```python
def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost
```

**输入：**
- `fea1`：[B, C, H, W] 参考图像特征
- `fea2`：[B, C, H, W] 目标图像特征
- `num_groups`：分组数量

**输出：**
- `cost`：[B, num_groups, H, W] 分组相关结果

**数据变换过程：**
```
fea1: [B, C, H, W]
fea2: [B, C, H, W]
    │
    ▼ [view]
fea1: [B, num_groups, channels_per_group, H, W]
fea2: [B, num_groups, channels_per_group, H, W]
    │
    ▼ [乘法]
fea1 * fea2: [B, num_groups, channels_per_group, H, W]
    │
    ▼ [mean(dim=2)]
cost: [B, num_groups, H, W]
```

**核心思想：** 将 C 个通道分成 num_groups 组，每组独立计算相关性后取平均

---

#### 1.4 build_gwc_volume（第 159-172 行）

**代码：**
```python
def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(
                refimg_fea[:, :, :, i:], 
                targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume
```

**输入：**
- `refimg_fea`：[B, C, H, W] 左图特征
- `targetimg_fea`：[B, C, H, W] 右图特征
- `maxdisp`：最大视差搜索范围（通常 192）
- `num_groups`：分组数量

**输出：**
- `volume`：[B, num_groups, maxdisp, H, W] 分组相关体积

**数据变换过程：**
```
对于每个视差值 d = 0, 1, 2, ..., maxdisp-1：
    │
    ├── d = 0：
    │   左图[:, :, :, :] 与 右图[:, :, :, :] 计算分组相关
    │
    └── d > 0：
        左图[:, :, :, d:] 与 右图[:, :, :, :-d] 计算分组相关
        存储到 volume[:, :, d, :, d:]

最终 volume: [B, num_groups, maxdisp, H, W]
```

**示意图：**
```
视差维度 (maxdisp)
    │
    ▼
┌─────────────────────────────┐
│ d=0  │ d=1  │ d=2  │ ...  │  ← 每层是 [B, num_groups, H, W]
│      │      │      │       │
└─────────────────────────────┘
       左图滑动方向 →
```

---

#### 1.5 disparity_regression（第 223-227 行）

**代码：**
```python
def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)
```

**输入：**
- `x`：[B, maxdisp, H, W] 视差概率分布（softmax 后的结果）

**输出：**
- `disp`：[B, 1, H, W] 期望视差值

**数学原理：**
$$\hat{d} = \sum_{d=0}^{D_{max}-1} d \cdot P(d)$$

**数据变换过程：**
```
x: [B, maxdisp, H, W]  # 概率分布，每像素是 maxdisp 维向量
    │
    ▼ [乘以视差值]
x * disp_values: [B, maxdisp, H, W]  # 每个通道乘以对应视差 0,1,2...
    │
    ▼ [sum(dim=1)]
disp: [B, 1, H, W]  # 期望值
```

---

#### 1.6 context_upsample（第 240-250 行）

**代码：**
```python
def context_upsample(disp_low, up_weights):
    b, c, h, w = disp_low.shape
        
    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    disp = (disp_unfold*up_weights).sum(1)
        
    return disp
```

**输入：**
- `disp_low`：[B, 1, H, W] 低分辨率视差图
- `up_weights`：[B, 9, 4H, 4W] 上采样权重

**输出：**
- `disp`：[B, 1, 4H, 4W] 上采样后的视差图

**数据变换过程：**
```
disp_low: [B, 1, H, W]
    │
    ▼ [unfold 3x3 卷积核]
disp_unfold: [B, 9, H, W]  # 9 个邻域块
    │
    ▼ [interpolate 4x]
disp_unfold: [B, 9, 4H, 4W]
    │
    ▼ [加权求和]
disp: [B, 1, 4H, 4W]
```

---

#### 1.7 FeatureAtt（submodule.py 第 252-260 行）

**代码：**
```python
class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1))

    def forward(self, cv, feat):
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att)*cv
        return cv
```

**输入：**
- `cv`：[B, cv_chan, H, W] 相关体积特征
- `feat`：[B, feat_chan, H, W] 辅助特征（如单目特征）

**输出：**
- `cv`：[B, cv_chan, H, W] 加权后的相关体积

**数据变换过程：**
```
feat: [B, feat_chan, H, W]
    │
    ▼ [feat_att: 1x1 conv]
feat_att: [B, cv_chan, H, W]
    │
    ▼ [sigmoid]
attention: [B, cv_chan, H, W]  # 每通道的注意力权重
    │
    ▼ [乘法]
cv: [B, cv_chan, H, W] * attention → [B, cv_chan, H, W]
```

---

#### 1.8 BasicConv_IN（第 90-101 行）

**代码：**
```python
class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)
        return x
```

**输入：**
- `x`：[B, C_in, H, W] 或 [B, C_in, D, H, W]（2D 或 3D）

**输出：**
- `x`：[B, C_out, H, W] 或 [B, C_out, D, H, W]

**与 BasicConv 的区别：**
| 归一化方式 | BasicConv | BasicConv_IN |
|-----------|-----------|--------------|
| 归一化层 | BatchNorm | InstanceNorm |
| 公式 | $y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$ | $y_{b,c,i,j} = \frac{x_{b,c,i,j} - \mu_{b,c}}{\sqrt{\sigma_{b,c}^2 + \epsilon}} \cdot \gamma + \beta$ |
| 作用域 | 批次维度 | 单样本 + 通道维度 |

**InstanceNorm vs BatchNorm：**
```
BatchNorm: 对整个批次做归一化 (B, C, H, W) → 均值/方差跨批次计算
InstanceNorm: 对每个样本每个通道独立归一化 (B, C, H, W) → 均值/方差仅在 H×W 内计算

InstanceNorm 更适合风格迁移等任务，因为每个样本的样式应该独立
```

---

#### 1.9 Conv2x_IN（第 104-136 行）

**代码：**
```python
class Conv2x_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv_IN(out_channels*2, out_channels*mul, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x
```

**输入：**
- `x`：[B, C_in, H, W]（需要下采样的特征）
- `rem`：[B, C_rem, H', W']（跳跃连接的特征）

**输出：**
- `x`：[B, C_out, H', W']

**与 Conv2x 的区别：**
- 使用 InstanceNorm (IN) 替代 BatchNorm (BN)
- 参数接口与 Conv2x 完全一致

---

#### 1.10 norm_correlation 与 build_norm_correlation_volume（第 159-172 行）

**代码：**
```python
def norm_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume
```

**norm_correlation 输入/输出：**
- `fea1`：[B, C, H, W] 参考特征
- `fea2`：[B, C, H, W] 目标特征
- 输出：`cost`：[B, 1, H, W] 归一化相关系数

**norm_correlation 数据变换过程：**
```
fea1: [B, C, H, W]
    │
    ▼ [L2 归一化: fea1 / ||fea1||]
fea1_norm: [B, C, H, W]
    │
    ▼ [乘法]
fea1_norm * fea2_norm: [B, C, H, W]
    │
    ▼ [mean(dim=1)]
cost: [B, 1, H, W]
```

**归一化相关性公式：**
$$\text{cost} = \frac{1}{C} \sum_{c=1}^{C} \frac{fea1_c \cdot fea2_c}{\|fea1_c\| \cdot \|fea2_c\|}$$

**build_norm_correlation_volume 输入/输出：**
- `refimg_fea`：[B, C, H, W] 左图特征
- `targetimg_fea`：[B, C, H, W] 右图特征
- `maxdisp`：最大视差搜索范围
- 输出：`volume`：[B, 1, maxdisp, H, W]

---

#### 1.11 correlation 与 build_correlation_volume（第 177-186 行）

**代码：**
```python
def correlation(fea1, fea2):
    cost = torch.sum((fea1 * fea2), dim=1, keepdim=True)
    return cost

def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume
```

**correlation 输入/输出：**
- `fea1`：[B, C, H, W]
- `fea2`：[B, C, H, W]
- 输出：`cost`：[B, 1, H, W] 逐通道点积之和

**correlation 数据变换过程：**
```
fea1: [B, C, H, W]
fea2: [B, C, H, W]
    │
    ▼ [逐元素乘法]
fea1 * fea2: [B, C, H, W]
    │
    ▼ [sum(dim=1)]
cost: [B, 1, H, W]
```

**相关性类型对比：**
| 类型 | 公式 | 输出通道 | 特点 |
|------|------|---------|------|
| correlation | $\sum_c fea1_c \cdot fea2_c$ | 1 | 原始点积，无归一化 |
| norm_correlation | $\frac{1}{C}\sum_c \frac{fea1_c \cdot fea2_c}{\|fea1_c\|\|fea2_c\|}$ | 1 | 归一化，值域 [-1, 1] |
| groupwise_correlation | $\frac{1}{C/g}\sum_{c \in g} fea1_c \cdot fea2_c$ | num_groups | 分组，保留结构信息 |

---

#### 1.12 build_concat_volume（第 191-201 行）

**代码：**
```python
def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume
```

**输入：**
- `refimg_fea`：[B, C, H, W] 左图特征
- `targetimg_fea`：[B, C, H, W] 右图特征
- `maxdisp`：最大视差搜索范围

**输出：**
- `volume`：[B, 2*C, maxdisp, H, W] 拼接相关体积

**数据变换过程：**
```
对于每个视差值 d = 0, 1, 2, ..., maxdisp-1：
    │
    ├── d = 0：
    │   volume[:, :C, 0, :, :] = refimg_fea      # 左图特征
    │   volume[:, C:, 0, :, :] = targetimg_fea  # 右图特征
    │
    └── d > 0：
        volume[:, :C, d, :, :] = refimg_fea[:, :, :, :]      # 完整左图
        volume[:, C:, d, :, d:] = targetimg_fea[:, :, :, :-d] # 裁剪右图

最终 volume: [B, 2*C, maxdisp, H, W]
```

**与其他关联体积的区别：**
| 类型 | 通道数 | 信息内容 |
|------|--------|---------|
| correlation | 1 | 标量相似度 |
| groupwise_correlation | num_groups | 分组向量相似度 |
| build_concat_volume | 2*C | 原始特征拼接（可学习） |

---

#### 1.13 Propagation（第 253-270 行）

**代码：**
```python
class Propagation(nn.Module):
    def __init__(self):
        super(Propagation, self).__init__()
        self.replicationpad = nn.ReplicationPad2d(1)

    def forward(self, disparity_samples):
        one_hot_filter = torch.zeros(5, 1, 3, 3, device=disparity_samples.device).float()
        one_hot_filter[0, 0, 0, 0] = 1.0
        one_hot_filter[1, 0, 1, 1] = 1.0
        one_hot_filter[2, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 2, 0] = 1.0
        one_hot_filter[4, 0, 0, 2] = 1.0
        disparity_samples = self.replicationpad(disparity_samples)
        aggregated_disparity_samples = F.conv2d(disparity_samples,
                                                 one_hot_filter, padding=0)
        return aggregated_disparity_samples
```

**输入：**
- `disparity_samples`：[B, 1, H, W] 视差样本

**输出：**
- `aggregated_disparity_samples`：[B, 5, H, W] 聚合后的视差

**卷积核可视化（5 个 3x3 位置）：**
```
one_hot_filter 权重排布：
filter[0]:  filter[1]:  filter[2]:  filter[3]:  filter[4]:
┌───┬───┬───┐ ┌───┬───┬───┐ ┌───┬───┬───┐ ┌───┬───┬───┐ ┌───┬───┬───┐
│ 1 │ 0 │ 0 │ │ 0 │ 0 │ 0 │ │ 0 │ 0 │ 1 │ │ 0 │ 0 │ 0 │ │ 0 │ 0 │ 0 │
├───┼───┼───┤ ├───┼───┼───┤ ├───┼───┼───┤ ├───┼───┼───┤ ├───┼───┼───┤
│ 0 │ 0 │ 0 │ │ 0 │ 1 │ 0 │ │ 0 │ 0 │ 0 │ │ 0 │ 0 │ 0 │ │ 0 │ 0 │ 0 │
├───┼───┼───┤ ├───┼───┼───┤ ├───┼───┼───┤ ├───┼───┼───┤ ├───┼───┼───┤
│ 0 │ 0 │ 0 │ │ 0 │ 0 │ 0 │ │ 0 │ 0 │ 0 │ │ 1 │ 0 │ 0 │ │ 0 │ 0 │ 0 │
└───┴───┴───┘ └───┴───┴───┘ └───┴───┴───┘ └───┴───┴───┘ └───┴───┴───┘
  左上(0,0)   中心(1,1)   右下(2,2)   左下(2,0)   右上(0,2)
```

**聚合方式：**
- 每个位置的视差由其 5 个对角线邻居聚合
- 使用 ReplicationPad2d 避免边界问题
- 输出通道 5 表示 5 个方向的聚合结果

---

#### 1.14 Propagation_prob（第 273-292 行）

**代码：**
```python
class Propagation_prob(nn.Module):
    def __init__(self):
        super(Propagation_prob, self).__init__()
        self.replicationpad = nn.ReplicationPad3d((1, 1, 1, 1, 0, 0))

    def forward(self, prob_volume):
        one_hot_filter = torch.zeros(5, 1, 1, 3, 3, device=prob_volume.device).float()
        one_hot_filter[0, 0, 0, 0, 0] = 1.0
        one_hot_filter[1, 0, 0, 1, 1] = 1.0
        one_hot_filter[2, 0, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 0, 2, 0] = 1.0
        one_hot_filter[4, 0, 0, 0, 2] = 1.0

        prob_volume = self.replicationpad(prob_volume)
        prob_volume_propa = F.conv3d(prob_volume, one_hot_filter, padding=0)
        return prob_volume_propa
```

**输入：**
- `prob_volume`：[B, 1, D, H, W] 视差概率体积

**输出：**
- `prob_volume_propa`：[B, 5, D, H, W] 聚合后的概率体积

**与 Propagation 的区别：**
| 维度 | Propagation | Propagation_prob |
|------|-------------|------------------|
| 输入 | [B, 1, H, W] | [B, 1, D, H, W] |
| 卷积 | 2D (3x3) | 3D (1x3x3) |
| 输出 | [B, 5, H, W] | [B, 5, D, H, W] |

**3D 卷积核排布：**
```
one_hot_filter 形状: [5, 1, 1, 3, 3]
           深度维度 (D)
              │
              ▼
filter[0]: ┌─────────┐
           │ 1 0 0   │
           │ 0 0 0   │  ← 每个 filter 在 H×W 平面上是 3x3 对角线
           │ 0 0 0   │
           └─────────┘

filter[1-4]: 类似，只是 1 的位置不同
```

**应用场景：**
- Propagation：对视差图进行空间传播（边缘保留平滑）
- Propagation_prob：对视差概率体积进行 3D 传播（跨视差维度）

---

### 2. core/utils/utils.py

#### 2.1 InputPadder（第 9-26 行）

**代码：**
```python
class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
```

**输入：**
- `dims`：(B, C, H, W) 或 (B, H, W) 图像尺寸
- `divis_by`：需要被整除的数（默认 8）

**输出：**
- `pad()` 返回填充后的图像
- `unpad()` 返回裁剪回原始尺寸的图像

**填充原理：**
```
原始尺寸 H × W
    │
    ▼ 计算需要填充的像素数
pad_ht = ceil(H/8) * 8 - H
pad_wd = ceil(W/8) * 8 - W
    │
    ▼ 填充模式
[左, 右, 上, 下]
    │
    ▼ F.pad(mode='replicate')
填充后尺寸: (H + pad_ht) × (W + pad_wd)
```

---

#### 2.2 bilinear_sampler（第 47-57 行）

**代码：**
```python
def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img
```

**输入：**
- `img`：[B, C, 1, W] 图像（或特征）
- `coords`：[B, 1, 1, W] 采样坐标 (x, y)

**输出：**
- 采样后的图像

**坐标变换：**
```
像素坐标: [0, W-1]
    │
    ▼ 归一化
grid: [-1, 1]
```

---

#### 2.3 coords_grid（第 60-63 行）

**代码：**
```python
def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)
```

**输入：**
- `batch`：批次大小
- `ht`：图像高度
- `wd`：图像宽度

**输出：**
- `coords`：[B, 2, H, W] 坐标网格

**数据变换过程：**
```
torch.arange(ht) → [0, 1, 2, ..., ht-1]  # y 坐标
torch.arange(wd) → [0, 1, 2, ..., wd-1]  # x 坐标
    │
    ▼ meshgrid
meshgrid(y, x) → [[y0,y1,...], [x0,x1,...]]
    │
    ▼ stack([x, y], dim=0)  # 注意顺序反转
coords[0] = x, coords[1] = y
    │
    ▼ [None].repeat(batch)
coords: [B, 2, H, W]
```

---

### 3. core/warp.py

#### 3.1 meshgrid（第 17-36 行）

**代码：**
```python
def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.size(1) == 3
    return grid
```

**输入：**
- `img`：[B, C, H, W] 任意张量（用于获取尺寸和设备）
- `homogeneous`：是否返回齐次坐标

**输出：**
- `grid`：[B, 2, H, W] 或 [B, 3, H, W]

**示意图：**
```
grid[:, 0, i, j] = j  # x 坐标 (列索引)
grid[:, 1, i, j] = i  # y 坐标 (行索引)

例如 4x3 图像：
    j=0  j=1  j=2
i=0 [0,0] [1,0] [2,0]
i=1 [0,1] [1,1] [2,1]
i=2 [0,2] [1,2] [2,2]
i=3 [0,3] [1,3] [2,3]
```

---

#### 3.2 normalize_coords（第 5-14 行）

**代码：**
```python
def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid
```

**输入：**
- `grid`：[B, 2, H, W] 像素坐标

**输出：**
- `grid`：[B, H, W, 2] 归一化坐标 [-1, 1]

**变换公式：**
$$x_{norm} = 2 \cdot \frac{x}{w-1} - 1$$
$$y_{norm} = 2 \cdot \frac{y}{h-1} - 1$$

---

#### 3.3 disp_warp（第 42-70 行）

**代码：**
```python
def disp_warp(img, disp, padding_mode='border'):
    """Warping by disparity
    Args:
        img: [B, 3, H, W]
        disp: [B, 1, H, W], positive
        padding_mode: 'zeros' or 'border'
    Returns:
        warped_img: [B, 3, H, W]
        valid_mask: [B, 3, H, W]
    """
    grid = meshgrid(img)  # [B, 2, H, W] in image scale
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    warped_img = interp(img, sample_grid, padding_mode)

    mask = torch.ones_like(img)
    valid_mask = interp(mask, sample_grid, padding_mode)
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1
    return warped_img, valid_mask
```

**输入：**
- `img`：[B, 3, H, W] 左图像（或特征）
- `disp`：[B, 1, H, W] 视差图

**输出：**
- `warped_img`：[B, 3, H, W] 扭曲后的图像
- `valid_mask`：[B, 3, H, W] 有效区域掩码

**数据变换过程：**
```
img: [B, 3, H, W]          左图
disp: [B, 1, H, W]         视差值 (正值)
    │
    ▼ meshgrid
grid: [B, 2, H, W]         像素坐标 (x, y)
    │
    ▼ offset = [-disp, 0]
offset: [B, 2, H, W]       偏移量
    │
    ▼ grid + offset
sample_grid: [B, 2, H, W]  目标像素位置
    │
    ▼ normalize_coords
sample_grid: [B, H, W, 2]  归一化到 [-1, 1]
    │
    ▼ F.grid_sample
warped_img: [B, 3, H, W]   扭曲后的"右图"
```

**核心原理：**
```
左图像素 (x, y) 的视差为 d
右图对应位置 = (x - d, y)

通过扭曲，用左图预测右图应该长什么样
```

---

## 第二阶段：特征提取与更新模块

---

### 4. core/extractor.py

#### 4.1 ResidualBlock（第 7-58 行）

**代码：**
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        # ... 其他 norm_fn 选项 ...

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)
```

**输入：**
- `x`：[B, in_planes, H, W]

**输出：**
- `y`：[B, planes, H/stride, W/stride]

**数据变换过程：**
```
x: [B, in_planes, H, W]
    │
    ▼ conv1 (stride)
y: [B, planes, H/s, W/s]
    │
    ▼ norm1
y: [B, planes, H/s, W/s]
    │
    ▼ relu
y: [B, planes, H/s, W/s]
    │
    ▼ conv2
y: [B, planes, H/s, W/s]
    │
    ▼ norm2
y: [B, planes, H/s, W/s]
    │
    ▼ relu
y: [B, planes, H/s, W/s]
    │
    ├─[stride != 1 or in != out]──▼ downsample
x: [B, planes, H/s, W/s] ──────────┘
    │
    ▼ x + y (残差连接)
out: [B, planes, H/s, W/s]
```

---

#### 4.2 BasicEncoder（第 130-185 行）

**代码：**
```python
class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, downsample=3):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        # ...

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))

        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

    def forward(self, x, dual_inp=False):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        # ...
        return x
```

**输入：**
- `x`：[B, 3, H, W] RGB 图像

**输出：**
- `x`：[B, output_dim, H/8, W/8]

**网络结构：**
```
输入图像: [B, 3, H, W]
    │
    ▼ conv1 (7x7, stride=1或2)
    │
    ▼ layer1: [64, H', W'] → [64, H', W']
    │
    ▼ layer2: [64, H', W'] → [96, H'/2, W'/2]
    │
    ▼ layer3: [96, H'/2, W'/2] → [128, H'/4, W'/4]
    │
    ▼ conv2 (1x1)
输出: [B, 128, H'/4, W'/4]
```

---

### 5. core/update.py

#### 5.1 ConvGRU（第 38-51 行）

**代码：**
```python
class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)
        h = (1-z) * h + z * q
        return h
```

**输入：**
- `h`：[B, hidden_dim, H, W] 上一时刻的隐藏状态
- `x_list`：当前时刻的输入特征列表

**输出：**
- `h`：[B, hidden_dim, H, W] 更新后的隐藏状态

**GRU 公式：**
$$z = \sigma(W_z * [h, x] + b_z) \quad \text{（更新门）}$$
$$r = \sigma(W_r * [h, x] + b_r) \quad \text{（重置门）}$$
$$q = \tanh(W_q * [r \odot h, x] + b_q) \quad \text{（候选隐藏状态）}$$
$$h' = (1-z) \odot h + z \odot q \quad \text{（新隐藏状态）}$$

---

#### 5.2 SepConvGRU（第 54-76 行）

**代码：**
```python
class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, *x):
        # horizontal
        x = torch.cat(x, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h
```

**输入：**
- `h`：[B, hidden_dim, H, W]
- `x`：多个输入特征的拼接

**输出：**
- `h`：[B, hidden_dim, H, W]

**分解卷积示意：**
```
标准卷积 (3x3)：
┌─────────────────┐
│ w00 w01 w02 w03 │  参数量: H × W × C²
│ w10 w11 w12 w13 │
│ w20 w21 w22 w23 │
│ w30 w31 w32 w33 │
└─────────────────┘

分解为水平 (1x5) + 垂直 (5x1)：
水平: ┌─────┐  参数量: 5H × C²
     │ h0~h4│

垂直: ┌v0┐      参数量: 5W × C²
     │v1│
     │v2│
     │v3│
     │v4│
     └───┘
```

---

#### 5.3 BasicMultiUpdateBlock（第 147-179 行）

**代码：**
```python
class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        # ...

    def forward(self, net, inp, corr=None, disp=None, iter04=True, iter08=True, iter16=True, update=True):
        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.grgru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp
```

**输入：**
- `net`：[net[0], net[1], net[2]] 多尺度隐藏状态
- `inp`：上下文特征
- `corr`：关联特征
- `disp`：当前视差估计

**输出：**
- `net`：更新后的隐藏状态
- `mask_feat_4`：上采样掩码特征
- `delta_disp`：视差更新量

**多尺度更新流程：**
```
        ┌──────────────────────────────────────┐
        │           net[2] (1/16 scale)        │
        │  gru16: net[2] = GRU(net[2], pool2x(net[1]))
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │           net[1] (1/8 scale)         │
        │  gru08: net[1] = GRU(net[1], pool2x(net[0]), ...)
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │           net[0] (1/4 scale)         │
        │  gru04: net[0] = GRU(net[0], motion_features, ...)
        └──────────────┬───────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   disp_head    │
              │ delta_disp = ? │
              └────────────────┘
```

---

### 6. core/geometry.py

#### 6.1 Combined_Geo_Encoding_Volume（第 5-68 行）

**代码：**
```python
class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []

        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)

    def __call__(self, disp, coords):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()
```

**输入：**
- `init_fmap1`, `init_fmap2`：[B, C, H, W] 左右图初始特征
- `geo_volume`：[B, C_geo, D, H, W] 几何编码体

**输出调用：**
- `disp`：[B, 1, H, W] 当前视差估计
- `coords`：[B, H, W] 像素坐标

**返回：**
- `out`：[B, feature_dim, H, W] 融合几何先验的关联特征

**构建过程：**
```
1. 初始化阶段：
   - 计算初始关联 init_corr
   - 构建 geo_volume 和 init_corr 的金字塔

2. 调用阶段：
   对于每个金字塔层级 i：
       - 根据当前视差 disp 生成采样偏移 dx
       - 用偏移后的坐标对几何编码体采样 → geo_volume_feat
       - 用原始坐标对初始关联采样 → init_corr_feat
       - 拼接两者
```

---

## 第三阶段：核心模型整合

---

### 7. core/refinement.py

#### 7.1 REMP（第 363-423 行）

**代码：**
```python
class REMP(nn.Module):
    def __init__(self):
        super(REMP, self).__init__()

        in_channels = 6
        channel = 32
        self.conv1_mono = conv2d(in_channels, 16)
        self.conv1_stereo = conv2d(in_channels, 16)
        self.conv2_mono = conv2d(1, 16)  # on low disparity
        self.conv2_stereo = conv2d(1, 16)  # on low disparity

        self.conv_start = BasicConv_now(64, channel, kernel_size=3, padding=2, dilation=2)

        self.RefinementBlock = Simple_UNet(in_channels=channel)

        self.AP = nn.AdaptiveAvgPool2d(1)
        self.LFE = nn.Sequential(
            nn.Conv2d(channel, channel * 2, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 2, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.LMC = nn.Sequential(
            default_conv(channel, channel, 3),
            default_conv(channel, channel * 2, 3),
            nn.ReLU(inplace=True),
            default_conv(channel * 2, channel, 3),
            nn.Sigmoid()
        )

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, disp_mono, disp_stereo, left_img, right_img):
        # 计算缺陷特征
        warped_right_mono = disp_warp(right_img, disp_mono)[0]
        flaw_mono = warped_right_mono - left_img

        warped_right_stereo = disp_warp(right_img, disp_stereo)[0]
        flaw_stereo = warped_right_stereo - left_img

        # 特征融合
        ref_flaw_mono = torch.cat((flaw_mono, left_img), dim=1)
        ref_flaw_stereo = torch.cat((flaw_stereo, left_img), dim=1)

        ref_flaw_mono = self.conv1_mono(ref_flaw_mono)
        ref_flaw_stereo = self.conv1_stereo(ref_flaw_stereo)

        disp_fea_mono = self.conv2_mono(disp_mono)
        disp_fea_stereo = self.conv2_stereo(disp_stereo)

        x = torch.cat((ref_flaw_mono, disp_fea_mono, ref_flaw_stereo, disp_fea_stereo), dim=1)
        x = self.conv_start(x)
        x = self.RefinementBlock(x)

        # 局部特征增强
        low = self.LFE(self.AP(x))
        motif = self.LMC(x)
        x = torch.mul((1 - motif), low) + torch.mul(motif, x)

        # 残差修正
        disp_stereo = nn.LeakyReLU()(disp_stereo + self.final_conv(x))

        return disp_stereo
```

**输入：**
- `disp_mono`：[B, 1, H, W] 单目深度图（低分辨率 4x 上采样后）
- `disp_stereo`：[B, 1, H, W] 立体匹配深度图
- `left_img`：[B, 3, H, W] 左图像
- `right_img`：[B, 3, H, W] 右图像

**输出：**
- `disp_stereo`：[B, 1, H, W] 精化后的视差图

**数据变换过程：**
```
左图 + 右图 + 单目视差 + 立体视差
    │
    ├── disp_warp(right_img, disp_mono) → warped_right_mono
    ├── warped_right_mono - left_img → flaw_mono (单目缺陷)
    │
    ├── disp_warp(right_img, disp_stereo) → warped_right_stereo
    ├── warped_right_stereo - left_img → flaw_stereo (立体缺陷)
    │
    ▼
特征拼接: [flaw_mono, left_img, flaw_stereo, disp_mono_fea, disp_stereo_fea]
    │
    ▼ conv_start
    │
    ▼ Simple_UNet (U-Net 精化)
    │
    ├── LFE: 全局注意力 → low
    ├── LMC: 局部调制 → motif
    └── x = (1-motif)*low + motif*x
    │
    ▼ final_conv
    │
    ▼ disp_stereo + residual
输出: 精化后的 disp_stereo
```

**缺陷（flaw）概念：**
```
如果视差估计正确：
    warped_right ≈ left_img
    flaw ≈ 0

如果视差估计错误：
    warped_right 与 left_img 有差异
    flaw ≠ 0  (反映了错误的方向和大小)
```

---

### 8. core/monster.py

#### 8.1 compute_scale_shift（第 23-57 行）

**代码：**
```python
def compute_scale_shift(monocular_depth, gt_depth, mask=None):
    flattened_depth_maps = monocular_depth.clone().view(-1).contiguous()
    sorted_depth_maps, _ = torch.sort(flattened_depth_maps)
    percentile_10_index = int(0.2 * len(sorted_depth_maps))
    threshold_10_percent = sorted_depth_maps[percentile_10_index]

    if mask is None:
        mask = (gt_depth > 0) & (monocular_depth > 1e-2) & (monocular_depth > threshold_10_percent)
    
    monocular_depth_flat = monocular_depth[mask]
    gt_depth_flat = gt_depth[mask]
    
    X = torch.stack([monocular_depth_flat, torch.ones_like(monocular_depth_flat)], dim=1)
    y = gt_depth_flat
    
    A = torch.matmul(X.t(), X) + 1e-6 * torch.eye(2, device=X.device)
    b = torch.matmul(X.t(), y)
    params = torch.linalg.solve(A, b)
    
    scale, shift = params[0].item(), params[1].item()
    
    return scale, shift
```

**输入：**
- `monocular_depth`：[H, W] 或 [B, H, W] 单目深度
- `gt_depth`：[H, W] 或 [B, H, W] 立体匹配深度

**输出：**
- `scale`：标度因子
- `shift`：偏移量

**数学原理：**

最小二乘法求解：
$$gt = s \cdot mono + t$$

通过正规方程求解：
$$\begin{bmatrix} s \\ t \end{bmatrix} = (X^T X)^{-1} X^T y$$

其中：
$$X = \begin{bmatrix} mono_1 & 1 \\ mono_2 & 1 \\ \vdots & \vdots \end{bmatrix}, \quad y = \begin{bmatrix} gt_1 \\ gt_2 \\ \vdots \end{bmatrix}$$

**有效掩码条件：**
- `gt_depth > 0`：GT 深度有效
- `monocular_depth > 1e-2`：单目深度非零
- `monocular_depth > threshold_10_percent`：去除单目深度过小的异常值

---

#### 8.2 hourglass（第 60-120 行）

**代码：**
```python
class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(
            BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(
            BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1))

        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, ...)
        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, ...)
        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, ...)

        self.agg_0 = nn.Sequential(...)
        self.agg_1 = nn.Sequential(...)

        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        # ...

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv
```

**输入：**
- `x`：[B, 8, D, H, W] 3D 相关体积（8 通道分组相关）
- `features`：[feat_4x, feat_8x, feat_16x, feat_32x] 单目特征金字塔

**输出：**
- `conv`：[B, 8, D, H, W] 精化后的几何编码体

**3D 卷积沙漏结构：**
```
输入: [B, 8, D, H, W]
    │
    ▼ conv1 (3D, stride=2)      D→D/2, H→H/2, W→W/2
    │
    ▼ feature_att_8 (融合 feat[1])
    │
    ▼ conv2 (3D, stride=2)      D→D/4, H→H/4, W→W/4
    │
    ▼ feature_att_16 (融合 feat[2])
    │
    ▼ conv3 (3D, stride=2)      D→D/8, H→H/8, W→W/8
    │
    ▼ feature_att_32 (融合 feat[3])
    │
    ▼ conv3_up (3D 反卷积)       D/8→D/4, H/8→H/4, W/8→W/4
    │
    ▼ cat([conv3_up, conv2])   通道拼接
    │
    ▼ agg_0
    │
    ▼ feature_att_up_16
    │
    ▼ conv2_up (3D 反卷积)       D/4→D/2, H/4→H/2, W/4→W/2
    │
    ▼ cat([conv2_up, conv1])
    │
    ▼ agg_1
    │
    ▼ feature_att_up_8
    │
    ▼ conv1_up (3D 反卷积)       D/2→D, H/2→H, W/2→W
    │
输出: [B, 8, D, H, W]
```

---

#### 8.3 Monster.forward（第 330-450 行）

**代码（核心流程）：**
```python
def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
    # 图像归一化
    image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
    image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
    
    # 1. 单目深度估计
    with torch.autocast(device_type='cuda', dtype=torch.float32): 
        depth_mono, features_mono_left, features_mono_right = self.infer_mono(image1, image2)

    # 2. 单目特征转换为立体匹配特征
    features_left = self.feat_transfer(features_mono_left)
    features_right = self.feat_transfer(features_mono_right)
    
    # 3. 双目特征提取
    stem_2x = self.stem_2(image1)
    stem_4x = self.stem_4(stem_2x)
    stem_8x = self.stem_8(stem_4x)
    stem_16x = self.stem_16(stem_8x)
    
    # 4. 构建几何编码体
    match_left = self.desc(self.conv(features_left[0]))
    match_right = self.desc(self.conv(features_right[0]))
    gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8)
    gwc_volume = self.corr_stem(gwc_volume)
    gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
    geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

    # 5. 初始化视差
    prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
    init_disp = disparity_regression(prob, self.args.max_disp//4)
    
    # 6. 多步迭代优化
    for itr in range(iters):
        disp = disp.detach()
        if itr >= 1:
            disp_mono_4x = disp_mono_4x.detach()
        geo_feat = geo_fn(disp, coords)
        
        if itr > iters - 8:  # 后 7 次迭代：融合单目信息
            if itr == iters - 7:
                # 尺度对齐
                scale, shift = compute_scale_shift(disp_mono_4x, disp)
                disp_mono_4x = scale * disp_mono_4x + shift
            
            # 计算缺陷特征
            warped_right_mono = disp_warp(features_right[0], disp_mono_4x)[0]
            flaw_mono = warped_right_mono - features_left[0]
            warped_right_stereo = disp_warp(features_right[0], disp)[0]
            flaw_stereo = warped_right_stereo - features_left[0]
            geo_feat_mono = geo_fn(disp_mono_4x, coords)
        
        # 混合更新
        if itr <= iters - 8:
            net_list, mask_feat_4, delta_disp = self.update_block(...)
        else:
            net_list, mask_feat_4, delta_disp = self.update_block_mix_stereo(...)
            net_list_mono, mask_feat_4_mono, delta_disp_mono = self.update_block_mix_mono(...)
            disp_mono_4x = disp_mono_4x + delta_disp_mono
        
        disp = disp + delta_disp
        
        if itr == iters - 1:  # 最后一次迭代
            refine_value = self.REMP(disp_mono_4x_up, disp_up, image1, image2)
            disp_up = disp_up + refine_value
        
        disp_preds.append(disp_up)

    return init_disp, disp_preds, depth_mono
```

**输入：**
- `image1`：[B, 3, H, W] 左图像
- `image2`：[B, 3, H, W] 右图像
- `iters`：迭代次数（默认 12）

**输出：**
- `init_disp`：初始视差
- `disp_preds`：每次迭代的视差预测列表
- `depth_mono`：单目深度

**完整数据流：**
```
左图 + 右图
    │
    ├─────────────────────────────────────────┐
    ▼                                         │
┌───────────────────────────────────────────┐ │
│        单目深度分支 (Depth Anything V2)    │ │
│                                           │ │
│ image → infer_mono → depth_mono           │ │
│                      → features_mono      │ │
└───────────────────────────────────────────┘ │
    │                                         │
    ▼                                         │
┌───────────────────────────────────────────┐ │
│           特征转换分支                      │ │
│                                           │ │
│ features_mono → feat_transfer → features │ │
│                                     ↓     │ │
│                               stereo feats │ │
└───────────────────────────────────────────┘ │
    │                                         │
    ▼                                         │
┌───────────────────────────────────────────┐ │
│           几何编码体构建                    │ │
│                                           │ │
│ stereo feats → match feat (desc)          │ │
│                    ↓                      │ │
│            build_gwc_volume → gwc_vol      │ │
│                    ↓                      │ │
│            cost_agg (hourglass) → geo_vol  │ │
│                    ↓                      │ │
│            classifier + softmax → prob      │ │
│                    ↓                      │ │
│            disparity_regression → init_disp│ │
└───────────────────────────────────────────┘ │
    │                                         │
    ▼                                         │
┌───────────────────────────────────────────┐ │
│           多步迭代优化 (12次)               │ │
│                                           │ │
│ 迭代 1-5: 仅立体匹配更新                   │ │
│   disp → update_block → delta_disp        │ │
│          disp = disp + delta_disp         │ │
│                                           │ │
│ 迭代 6-12: 立体 + 单目联合更新             │ │
│   尺度对齐: compute_scale_shift            │ │
│   disp_mono = scale * disp_mono + shift   │ │
│   缺陷特征: disp_warp → flaw              │ │
│   联合更新: update_block_mix_*             │ │
└───────────────────────────────────────────┘ │
    │                                         │
    ▼                                         │
┌───────────────────────────────────────────┐ │
│           REMP 精化                         │ │
│                                           │ │
│ disp_mono_up, disp_stereo_up → REMP → disp│ │
└───────────────────────────────────────────┘ │
    │                                         │
    ▼                                         │
输出: [init_disp, disp_preds, depth_mono]
```

---

## 附录：关键公式

### 视差与深度关系
$$d = \frac{f \cdot B}{Z}$$

### 期望视差回归
$$\hat{d} = \sum_{d=0}^{D_{max}} d \cdot P(d)$$

### Leaky ReLU
$$f(x) = \max(0, x) + \alpha \cdot \min(0, x)$$

### Smooth L1 Loss
$$\text{SmoothL1}(x) = \begin{cases} 0.5 x^2 & |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

---

## 文件索引表

| 文件 | 行号 | 类/函数 | 功能 |
|------|------|---------|------|
| submodule.py | 7-30 | BasicConv | 基础 2D/3D 卷积块 |
| submodule.py | 33-67 | Conv2x | U-Net 跳跃连接模块 |
| submodule.py | 90-101 | BasicConv_IN | InstanceNorm 版本 |
| submodule.py | 104-136 | Conv2x_IN | InstanceNorm 版本 |
| submodule.py | 150-156 | groupwise_correlation | 分组相关计算 |
| submodule.py | 159-172 | build_gwc_volume | 构建分组相关体积 |
| submodule.py | 159-172 | norm_correlation | 归一化相关性计算 |
| submodule.py | 159-172 | build_norm_correlation_volume | 构建归一化相关体积 |
| submodule.py | 177-186 | correlation | 原始点积相关计算 |
| submodule.py | 177-186 | build_correlation_volume | 构建点积相关体积 |
| submodule.py | 191-201 | build_concat_volume | 构建拼接相关体积 |
| submodule.py | 223-227 | disparity_regression | 视差回归 |
| submodule.py | 240-250 | context_upsample | 上下文上采样 |
| submodule.py | 252-260 | FeatureAtt | 通道注意力 |
| submodule.py | 253-270 | Propagation | 视差空间传播 |
| submodule.py | 273-292 | Propagation_prob | 概率体积 3D 传播 |
| utils.py | 9-26 | InputPadder | 图像尺寸填充 |
| utils.py | 47-57 | bilinear_sampler | 双线性采样 |
| utils.py | 60-63 | coords_grid | 坐标网格生成 |
| warp.py | 17-36 | meshgrid | 像素坐标网格 |
| warp.py | 42-70 | disp_warp | 视差扭曲 |
| extractor.py | 7-58 | ResidualBlock | 残差块 |
| extractor.py | 130-185 | BasicEncoder | 特征编码器 |
| update.py | 38-51 | ConvGRU | 卷积 GRU |
| update.py | 54-76 | SepConvGRU | 分离卷积 GRU |
| update.py | 147-179 | BasicMultiUpdateBlock | 多尺度更新块 |
| geometry.py | 5-68 | Combined_Geo_Encoding_Volume | 几何编码体 |
| refinement.py | 363-423 | REMP | 深度图精化 |
| monster.py | 23-57 | compute_scale_shift | 尺度对齐 |
| monster.py | 60-120 | hourglass | 3D 沙漏网络 |
| monster.py | 330-450 | Monster.forward | 主模型前向 |