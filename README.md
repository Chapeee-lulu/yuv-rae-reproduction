# YUV可逆对抗攻击复现

本项目复现论文 *Efficient and Transferable Reversible Adversarial Attacks Utilizing YUV Color Space* 的主要算法流程，包括RGB/YUV转换、ImageNet标签生成、Y通道攻击、模型集成、CAM区域约束、算术编码、PEE嵌入、可逆对抗样本生成与指标计算。

代码按照Step 01—10组织。默认入口对应复现主路径；`_cmp`表示补充实现对比；`step10_pro.py`是在不替换串行入口的前提下提供的批量并行版本。完整运行记录、结果口径和已知限制见`Record.md`。

## 项目结构

```text
Yuv_Reproduction/
├─ script/
│  ├─ step01.py
│  ├─ step02.py
│  ├─ step03.py
│  ├─ step04.py
│  ├─ step05.py
│  ├─ step06.py
│  ├─ step07.py
│  ├─ step08.py
│  ├─ step09.py
│  ├─ step10.py
│  └─ step10_pro.py
├─ src/yuv_reproduction/       # 核心算法与_cmp对比实现
├─ data/val/imgn_labels.csv    # 外部验证标签
├─ README.md
├─ Record.md
└─ Requirements.txt
```

仓库不包含ImageNet原图、模型权重、虚拟环境、运行缓存、批量输出以及自动测试文件。

## 环境

Python 3.11。GPU批量实验需要CUDA环境；Step 01—09中的部分数据处理步骤可以在CPU上运行。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\Requirements.txt
```

Torchvision预训练模型会从本地缓存读取；本地不存在权重时需要联网下载。

## 数据准备

将1000张299×299的RGB PNG图片放入：

```text
data/original_img/
├─ 0001.png
├─ 0002.png
├─ ...
└─ 1000.png
```

`data/val/imgn_labels.csv`保存外部验证标签和作者原文件名，只用于Step 03核对生成标签。Step 03根据InceptionV3、GoogLeNet和DenseNet161的一致预测生成`results/step03_labels.csv`；Step 04—10读取该生成标签，不直接读取外部验证标签。

## 顺序运行

在项目根目录依次运行：

```powershell
.\.venv\Scripts\python.exe .\script\step01.py
.\.venv\Scripts\python.exe .\script\step02.py
.\.venv\Scripts\python.exe .\script\step03.py
.\.venv\Scripts\python.exe .\script\step04.py
.\.venv\Scripts\python.exe .\script\step05.py
.\.venv\Scripts\python.exe .\script\step06.py
.\.venv\Scripts\python.exe .\script\step07.py
.\.venv\Scripts\python.exe .\script\step08.py
.\.venv\Scripts\python.exe .\script\step09.py
.\.venv\Scripts\python.exe .\script\step10.py
```

各步骤作用如下：

| 步骤 | 内容 |
| --- | --- |
| Step 01 | 检查运行环境、图片数量、格式与命名 |
| Step 02 | 实现RGB与YUV 4:4:4转换 |
| Step 03 | 生成并核对ImageNet标签 |
| Step 04 | 实现五种Y通道攻击 |
| Step 05 | 实现三模型等权logits集成 |
| Step 06 | 生成CAM显著区域掩膜 |
| Step 07 | 实现算术编码与PEE可逆嵌入 |
| Step 08 | 生成RAE并执行恢复流程 |
| Step 09 | 计算PSNR、SSIM、CIEDE2000、ASR与恢复指标 |
| Step 10 | 运行YUV、YACK、ENS、OURS消融流程 |

## Step 10并行加速

`step10.py`保留串行入口；`step10_pro.py`通过GPU批处理和CPU多进程加速1000张正式实验，并使用独立结果文件和断点，不覆盖串行结果。

```powershell
.\.venv\Scripts\python.exe .\script\step10_pro.py --sample-count 1000 --method all --attack all --epsilon-pixels 2 4 --max-outer-iterations 50 --saved-artifact-count 0 --gpu-batch-size 2 --cpu-workers 4
```

中断后使用相同参数并增加`--resume`。显存不足时程序会自动降低GPU批次；也可以手动调整`--gpu-batch-size`与`--cpu-workers`。不同批次和进程数可能造成少量数值边界差异，因此跨方法比较时应同时记录运行参数。

## `_cmp`补充路径

`_cmp`代码用于说明预处理、Y通道更新、CAM缩放和RGB恢复等实现选择对结果的影响，不属于论文主复现路径，也不替换默认算法。

Step 03可通过以下命令启用ImageNet归一化对比：

```powershell
.\.venv\Scripts\python.exe .\script\step03.py --normalized-cmp
```

其余`_cmp`函数保留在对应核心模块中，便于阅读和单独调用。`outputs/cmp/`与主复现目录隔离，对比结果不计入论文主结果。

## 输出目录

- `results/`：各步骤生成的CSV与JSON结果。
- `outputs/reproduction/`：复现主路径图片。
- `outputs/cmp/`：`_cmp`补充对比图片。
- `outputs/sup/`：工程补充产物。

GitHub仅保留Step 01—10正式实验的CSV/JSON结果；生成图片、断点文件、测试小样和报告派生数据不上传。正式结果与结论边界以`Record.md`中的实际记录为准。

## 已知边界

- ASR与视觉指标需要同时结合成功生成数量理解，不能忽略容量失败样本。
- YUV、YACK与ENS、OURS使用的白盒模型集合不同，跨方法迁移比较应统一使用共同黑盒模型。
- Pro版本保留加速入口，但不同batch/workers下的耗时不能直接解释为算法速度优势。
- 当前工程不将普通RGB文件路径表述为已经实现逐像素无损恢复。
- RDH全量基线和额外未完成实验不作为本仓库已复现结果。
