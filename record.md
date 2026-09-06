# YUV可逆对抗攻击复现记录

## Python文件用途与论文对应关系

| 我们的Python文件 | 用途 | 论文对应 | 作者代码参考 | 我们的独立复现 | 公式入口 |
|---|---|---|---|---|---|
| `src/yuv_rae/yuv_conversion.py` | RGB与YUV转换；区分浮点转换和整数取整 | 第3.1节，公式(13)(14) | `utils.py:3,20`；`atk.py:5,16` | 仅依据论文公式重写，不导入作者模块 | [`docs/paper_formulas/equations.md`](docs/paper_formulas/equations.md) |
| `src/yuv_rae/metrics.py` | 检查两张RGB图是否逐像素一致 | 论文Fig.6报告恢复图SSIM=1.000；逐像素检查为我们的补充 | `calMetrics.py:5`仅计算SSIM、PSNR、L2、L∞ | 增加完全相等、最大误差、平均误差和变化计数 | 本文件无论文公式 |
| `scripts/step_01_color_roundtrip.py` | 读取真实图像，分别执行浮点和整数往返实验，保存JSON结果 | 验证公式(13)(14)的数值行为 | 作者主程序在生成RAE前后调用颜色转换 | 将单一问题拆成可重复命令，不运行攻击 | [`docs/paper_formulas/equations.md`](docs/paper_formulas/equations.md) |
| `tests/test_yuv_conversion.py` | 自动检查典型颜色、误差上限和错误输入 | 非论文内容，属于复现工程质量控制 | 作者仓库没有对应单元测试 | 用最小样例验证独立实现 | 本文件无论文公式 |
| `src/yuv_rae/__init__.py` | 暴露当前阶段允许外部使用的函数 | 非论文内容 | 无直接对应 | 统一模块入口 | 本文件无论文公式 |

### 关于逐像素检查量的来源

我们当前记录的五项不是论文提出的一组“恢复无损指标”，而是为了审计无损性自行组合的检查量：

| 检查量 | 来源与作用 |
|---|---|
| `exactly_equal` | NumPy逐元素完全相等判断；最直接回答是否逐像素无损 |
| `max_absolute_error` | 常见数值误差检查；确认最坏像素通道相差多少 |
| `mean_absolute_error` | 标准平均绝对误差MAE；反映整体平均偏差 |
| `changed_channel_values` | 我们增加的诊断计数；统计多少个R/G/B通道值变化 |
| `changed_pixels` | 我们增加的诊断计数；统计多少个像素至少一个通道变化 |

论文用于评价RAE视觉质量的是PSNR、SSIM和CIEDE2000；作者代码`calMetrics.py`还计算L2和L∞。论文在Fig.6用SSIM=1.000说明恢复效果，但我们额外使用逐像素检查，避免只凭四舍五入显示的SSIM判断“error-free”。

