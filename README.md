# YUV 可逆对抗攻击：独立复现项目

这个目录用于从零复现论文 *Efficient and transferable reversible adversarial attacks utilizing YUV color space*。

## 原则

1. 论文 PDF 和作者开源仓库保持只读。
2. 本项目不导入作者仓库中的 Python 模块，也不直接复制其实现。
3. 每一步先写清输入、输出和验证条件，再进入下一步。
4. 先用 1 张图验证，再扩展到 10 张、100 张和完整数据集。

## 目录

```text
yuv_rae_reproduction/
├─ README.md                     当前说明
├─ PROJECT_STATUS.md             七阶段进度和验收条件
├─ LEARNING_LOG.md               代码、思路、验证与理解检查
├─ requirements-step01.txt       第1步所需依赖
├─ notes/
│  └─ step01_color_space.md      RGB/YUV转换原理与验证问题
│  └─ STEP_TEMPLATE.md           后续每一步统一记录模板
├─ src/
│  └─ yuv_rae/
│     ├─ __init__.py
│     ├─ yuv_conversion.py       RGB与YUV转换，数据保持4:4:4
│     └─ metrics.py              逐像素误差统计
├─ scripts/
│  └─ step_01_color_roundtrip.py 第1步可执行实验
├─ tests/
│  └─ test_yuv_conversion.py     第1步单元测试
└─ outputs/                      实验输出，不存放源数据
```

论文公式集中存放在`docs/paper_formulas/equations.md`，每个Python文件与论文、作者代码及公式的对应关系集中记录在`record.md`。

## 第1步：验证RGB与YUV转换

在 PowerShell 中执行：

```powershell
cd D:\Ecnu\yuv_rae_reproduction
python -m pip install -r requirements-step01.txt
python -m unittest discover -s tests -v
python scripts\step_01_color_roundtrip.py `
  --input "D:\Ecnu\Efficient-and-Transferable-Reversible-Adversarial-Attacks-Utilizing-YUV-Color-Space-main\ORI_IMG\0001_321.png"
```

脚本会输出：

- RGB→YUV→RGB后是否逐像素完全一致；
- 最大绝对像素误差；
- 平均绝对误差；
- 发生变化的像素数和比例；
- 原图、往返转换图和放大后的差异图。

第1步不是对抗攻击实验。它只回答一个基础问题：论文给出的RGB/YUV公式在整数图像上是否能够直接保证逐像素无损。

## 数据和官方代码的位置

- 论文：`D:\Ecnu\Efficient-and-transferable-reversible-adversarial-attacks-ut_2025_Neurocompu.pdf`
- 作者仓库：`D:\Ecnu\Efficient-and-Transferable-Reversible-Adversarial-Attacks-Utilizing-YUV-Color-Space-main`

两者仅作为论文依据、测试数据来源和结果对照，不属于本项目代码。

## GitHub同步方式

每完成一个可验证阶段后提交一次，不把多个阶段混在同一提交中：

```text
step01: verify RGB-YUV roundtrip
step02: reproduce ImageNet inference preprocessing
step03: implement constrained Y-channel FGSM
...
```

每次提交前必须同时更新：

1. 本阶段代码；
2. 自动化测试；
3. `LEARNING_LOG.md`中的作者代码参考、独立复现和实验结果；
4. 你的理解检查状态。

体积较小的`metrics.json`会同步到GitHub，便于复核实验数据；生成的图片默认只保存在本地，避免仓库被大量中间文件占满。
