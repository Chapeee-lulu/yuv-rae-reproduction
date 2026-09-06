# YUV可逆对抗攻击复现记录

## Python文件用途与论文对应关系

| 我们的Python文件 | 用途 | 论文对应 | 作者代码参考 | 我们的独立复现 | 公式入口 |
|---|---|---|---|---|---|
| `src/yuv_rae/yuv_conversion.py` | RGB与YUV转换；区分浮点转换和整数取整 | 第3.1节，公式(13)(14) | `utils.py:3,20`；`atk.py:5,16` | 仅依据论文公式重写，不导入作者模块 | [`docs/paper_formulas/equations.md`](docs/paper_formulas/equations.md) |
| `src/yuv_rae/pixel_comparison.py` | 检查两张RGB图是否逐像素一致 | 论文Fig.6报告恢复图SSIM=1.000；逐像素检查为我们的补充 | `calMetrics.py:5`仅计算SSIM、PSNR、L2、L∞ | 增加完全相等、RGB单通道最大/平均误差和变化计数 | 本文件无论文公式 |
| `scripts/step_01_color_roundtrip.py` | 读取真实图像，分别执行浮点和整数往返实验，保存JSON结果 | 验证公式(13)(14)的数值行为 | 作者主程序在生成RAE前后调用颜色转换 | 将单一问题拆成可重复命令，不运行攻击 | [`docs/paper_formulas/equations.md`](docs/paper_formulas/equations.md) |
| `tests/test_yuv_conversion.py` | 自动检查典型颜色、误差上限和错误输入 | 非论文内容，属于复现工程质量控制 | 作者仓库没有对应单元测试 | 用最小样例验证独立实现 | 本文件无论文公式 |
| `tests/test_pixel_comparison.py` | 验证“变化通道值数”和“变化像素数”含义不同 | 非论文内容，属于复现工程质量控制 | 作者仓库没有对应测试 | 构造2个像素的人工例子验证计数 | 本文件无论文公式 |
| `src/yuv_rae/__init__.py` | 暴露当前阶段允许外部使用的函数 | 非论文内容 | 无直接对应 | 统一模块入口 | 本文件无论文公式 |

### 文件命名说明

`metric`是“评价指标或测量量”，`metrics.py`并不是一种Python前缀。原文件只做RGB逐像素比较，叫`metrics.py`范围过大，因此改为更具体的`pixel_comparison.py`。

不把正式功能文件命名为`test_pixel.py`：Python项目通常把`test_*.py`留给自动化测试，测试工具也会按这个规则自动发现测试文件。现在的分工是：

- `src/yuv_rae/pixel_comparison.py`：真正被实验脚本调用的比较功能；
- `tests/test_pixel_comparison.py`：用人工小样例验证比较功能是否算对。

### 论文的5个整体实验指标

论文第4.1节明确列出五个主要评价指标：

| 评价对象 | 论文指标 |
|---|---|
| RAE视觉质量 | PSNR、SSIM、CIEDE2000 |
| RAE攻击能力 | White-box ASR、Black-box ASR |

这五个指标用于评价“生成的RAE质量和攻击能力”，不是五个“恢复无损指标”。Step 01还没有生成RAE，也没有执行攻击，所以现在不计算这五项；后续实验评价模块将逐项实现。

### 我们的逐像素检查量

下面这些不是论文的五项指标，而是为了回答“颜色转换后RGB是否逐像素一致”而增加的检查量：

| 检查量 | 来源与作用 |
|---|---|
| `rgb_arrays_exactly_equal` | NumPy逐元素完全相等判断；直接回答是否逐像素无损 |
| `max_rgb_channel_absolute_error` | 最坏的一个R/G/B通道值相差多少 |
| `mean_rgb_channel_absolute_error` | 所有R/G/B通道值的平均绝对误差 |
| `changed_rgb_channel_value_count` | 诊断误差在三个颜色通道中出现了多少次 |
| `changed_rgb_pixel_count` | 诊断误差在图像空间中影响了多少个像素位置 |

严格判定无损只需要`rgb_arrays_exactly_equal=True`，等价于`max_rgb_channel_absolute_error=0`。最后两个只是帮助分析误差分布，可以不用于最终结论。

例如两个像素从`[[0,0,0], [0,0,0]]`变成`[[1,0,0], [1,1,1]]`：共有4个RGB通道值发生变化，但只影响2个像素位置。因此两个计数回答的问题不同。

论文用于评价RAE视觉质量的是PSNR、SSIM和CIEDE2000；作者代码`calMetrics.py`还计算L2和L∞。论文在Fig.6用SSIM=1.000说明恢复效果，但我们额外使用逐像素检查，避免只凭四舍五入显示的SSIM判断“error-free”。
