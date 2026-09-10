# YUV可逆对抗样本复现记录

## 当前正式实验结果汇总

> 实验范围：每种方法均为5种攻击×2个ε，共10组配置；每组输入1000张，以下“生成数”为10组配置累计的成功张次，不是不同图片数量。<br>OURS：9382/10000，单组856–988，配置总耗时6.30h；YACK：9672/10000，单组886–998，配置总耗时4.43h；ENS：8668/10000，单组800–921，配置总耗时4.84h；YUV：9084/10000，单组816–967，配置总耗时1.94h。<br>CSV固定保存1、10、1000三个检查点；每个方法150行＝3个检查点×10组配置×5个评价模型。正式Table只读取`sample_count=1000`的50行。<br>PSNR、SSIM、CIEDE2000和ASR均以成功生成RAE的`generated_count`为统计范围，不能将CSV中的ASR直接解释为1000张端到端成功率；最终表格必须同时展示`generated_count`。<br>Table 5的模型角色按攻击来源区分：YUV、YACK只以InceptionV3为白盒，另外四个模型按迁移模型统计；ENS、OURS以InceptionV3、GoogLeNet、DenseNet161为白盒，ResNet50、VGG19为黑盒。原始CSV中的固定`model_role`不直接用于YUV/YACK聚合，Step 11按实际攻击模型重新标记。<br>四种方法的成功样本均通过内存中的精确恢复检查，但YACK、ENS、YUV运行时使用`saved_artifact_count=0`，没有进行全量落盘PNG恢复；不能表述为全量文件恢复。只有OURS＋MI-FGSM＋2/255的978张主路径和22张容量补齐样本完成1000/1000自包含PNG恢复。<br>并行配置不同：OURS为batch 2/workers 4，YACK为1/2，ENS和YUV为3/6。`0280.png`的ResNet50 CAM类别在OURS中为68，在其余三组中为66；作者路径使用ResNet50当前预测类别生成CAM，该差异属于批次推理边界差异，跨方法比较时必须备注。<br>ENS的FGSM＋2/255，以及YUV的FGSM、I-FGSM、PGD、MI-FGSM＋4/255，生成阶段保存的Inception预测与最终批量评价各相差1张；最终Table统一采用CSV中的五模型批量评价结果。<br>RDH只证明入口和恢复闭环可运行；全量实验已明确放弃。历史`_cmp`结果不纳入最终论文复现对比。后续不再运行任何未完成实验，只在报告中标明状态。

| 步骤 | 步骤名称 | 所在py | 所在函数 | 函数作用 | 目的 | 对应原文编号 | 结果 | 复现效果 | 可以补充 | 补充效果 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 环境和数据 | `s0101_env.py` | `collect_runtime_environment()` | 检查环境和CUDA |  |  | Python 3.11.9<br>PyTorch 2.7.1+cu128<br>CUDA 12.8<br>RTX 5060 | 新增 | 计算CUDA张量 |  |  |
|  |  | `s0102_data.py` | `LocalImageDatasetReport.to_dictionary()` | 转换为字典 |  |  | 环境与数据报告成功序列化 | 新增 |  |  |  |
|  |  |  | `val_png()` | 检查数量、格式、命名和可读性 |  |  | 1000张PNG均可读<br>RGB<br>299×299 | 新增 |  |  |  |
|  |  | `step01.py` | `main()` | 执行step01 |  |  | 生成`results/step01.json`<br>3项测试通过 | Step 01复现测试通过 |  |  | 环境和无标签原图满足后续复现条件 |
| 2 | RGB与YUV转换 | `s02_rgb_yuv.py` | `_val_three_channel_image()` | 检查输入形状 |  | 公式13、公式14 | 5种人工颜色和10张真实图片形状检查通过 | 通过 |  |  |  |
|  |  |  | `cal_rgb_to_yuv()` | 公式13计算浮点YUV |  |  |  | 通过 |  |  |  |
|  |  |  | `cal_yuv_to_rgb()` | 公式14计算浮点RGB |  |  |  | 最大通道误差1 |  |  |  |
|  |  |  | `round_color_channel_values()` | 取整 |  |  | 按作者路径保留整数YUV<br>红色V和蓝色U可取到256 | 通过 |  |  |  |
|  |  |  | `clip_color_channel_values()` | 裁剪，便于uint8 |  |  | 仅在保存预览图时裁剪到0–255 | 通过 |  |  |  |
|  |  |  | `convert_color_channels_to_uint8()` | 转为uint8 |  |  |  | 通过 |  |  |  |
|  |  |  | `rgb_to_yuv()` | RGB转YUV，没转uint8，可能出现256 |  |  |  | 通过 |  |  |  |
|  |  |  | `yuv_to_rgb()` | YUV转RGB，保留浮点数 |  |  |  | 最大通道误差1<br>视觉无明显差异 |  |  |  |
|  |  | `step02.py` | `cal_round_trip_error()` | 统计往返转换误差 |  |  | 首张图片往返平均误差0.3046<br>相等通道比例69.54%<br>人工颜色平均误差0.0667<br>10张真实图片平均误差0.2452 | 最大通道误差均为1 |  |  |  |
|  |  |  | `summarize_yuv_channels()` | 统计YUV通道 |  |  |  | 通过 |  |  |  |
|  |  |  | `check_one_image()` | 检查单张图片 |  |  |  | 最大通道误差1 |  |  |  |
|  |  |  | `main()` | 执行第2步检查 |  |  | 生成`results/step02.json`<br>保存Y/U/V及恢复RGB共4张预览图<br>3项测试通过 | Step 02主复现通过 |  |  | 公式13和公式14的整数取整路径已复现<br>往返误差不超过1个灰度级 |
| **2补充** | **对比路径** |  | `rgb_to_yuv_cmp()` | 对比：裁剪并转uint8 | 与主复现路径对比 |  | 历史测试：3/3通过<br>5种人工颜色和10张真实图片 |  | 算法或实现对比 | 人工边界颜色：主路径可为256且保持float32，对比路径裁剪到255并转uint8<br>10张真实图片：越界值0个，两条路径YUV数值差异0个 | 对当前10张真实图片没有数值影响<br>两条路径平均RGB误差均为0.245240、最大误差均为1 |
|  |  |  | `yuv_to_rgb_cmp()` | 对比：转uint8 |  |  |  |  | 算法或实现对比 | 两条路径恢复数值一致<br>对比路径固定输出uint8 | 主路径保留更宽的中间值范围，适合后续可逆处理 |
| 3 | ImageNet推理 | `s03_imgn_inference.py` | `load_imgn_val_labels()` | 读取外部验证集标签 |  |  |  |  |  |  |  |
|  |  |  | `generate_imgn_labels()` | 根据三个白盒模型一致预测生成标签 |  |  | 三个白盒模型对1000张图片全部预测一致<br>生成1000个标签 | `step03_labels.csv`：保存图片名、生成标签、验证标签、白盒一致标记和标签匹配标记<br>作为后续步骤读取的验证后生成标签 |  |  |  |
|  |  |  | `load_generated_imgn_labels()` | 读取Step 03生成且验证通过的标签 |  |  |  |  |  |  |  |
|  |  |  | `convert_image_to_unit_tensor()` | 转为0到1张量 |  |  |  |  |  |  |  |
|  |  |  | `imgn_input()` | 生成非归一化输入[默认直接返回] |  |  |  |  |  |  |  |
|  |  |  | `add_batch_dimension()` | 增加批次维度 |  |  |  |  |  |  |  |
|  |  |  | `ImgnPngDataset.__init__()` | 保存不含标签的原始图片路径 |  |  |  |  |  |  |  |
|  |  |  | `ImgnPngDataset.__len__()` | 返回图片数量 |  |  |  |  |  |  |  |
|  |  |  | `ImgnPngDataset.__getitem__()` | 读取无标签图片 |  |  |  |  |  |  |  |
|  |  |  | `load_imgn_model()` | 加载预训练模型 |  |  |  |  |  |  |  |
|  |  |  | `predict_class_indices()` | 预测类别编号，取logits的argmax |  |  |  |  |  |  |  |
|  |  |  | `predict_imgn_dataset()` | 预测数据集 |  |  | InceptionV3：1000/1000，4.39s<br>GoogLeNet：1000/1000，3.33s<br>DenseNet161：1000/1000，8.60s<br>ResNet50：797/1000，4.99s<br>VGG19：667/1000，8.16s | `step03_predictions.csv`：保存每张图片的5个模型预测、生成标签、验证标签及一致性和正确性标记 |  |  |  |
|  |  | `step03.py` | `parse_arguments()` | 选择是否归一化处理 |  |  |  |  |  |  |  |
|  |  |  | `main()` | 执行第3步检查 |  |  |  |  |  |  |  |
| **3补充** | **对比路径** |  | `imgn_input_cmp()` | 生成归一化输入[对比] | 与主复现路径对比 |  | 历史测试：3/3通过<br>1000张图片、5个模型、两种输入 |  | 算法或实现对比 | 未归一化/归一化正确数：InceptionV3 1000/991，GoogLeNet 1000/981，DenseNet161 1000/989，ResNet50 797/977，VGG19 667/967<br>预测改变数依次为9、19、11、202、331 | 归一化提高两个黑盒模型正确率，但使三个白盒共同正确数从1000降为964<br>主复现继续使用不归一化输入 |
| 4 | Y通道攻击 | `s04_y_attack.py` | `rgb_to_yuv_tensor()` | 公式13 |  | Algorithm 1 |  |  |  |  |  |
|  |  |  | `yuv_to_rgb_tensor()` | 公式14 |  |  |  |  |  |  |  |
|  |  |  | `cal_yuv_grad()` | 复现：计算完整YUV梯度 |  |  |  |  |  |  |  |
|  |  |  | `clip_rgb_update()` | 复现：裁剪RGB扰动 |  |  |  |  |  |  |  |
|  |  |  | `build_attack_result()` | 复现：返回结果 |  |  |  |  |  |  |  |
|  |  |  | `yfgsm_attack()` | 复现：生成YFGSM |  |  |  |  |  |  |  |
|  |  |  | `yifgsm_attack()` | 复现：生成YI-FGSM |  |  |  |  |  |  |  |
|  |  |  | `ypgd_attack()` | 复现：生成YPGD |  |  |  |  |  |  |  |
|  |  |  | `ymifgsm_attack()` | 复现：生成YMI-FGSM |  |  |  |  |  |  |  |
|  |  |  | `ynifgsm_attack()` | 复现：生成YNI-FGSM |  |  |  |  |  |  |  |
|  |  | `step04.py` | `main()` | 用一张真实图片验证五种攻击 |  |  | `0001.png`，标签321<br>ε=2/255：YI-FGSM、YPGD、YMI-FGSM攻击成功，3/5<br>ε=4/255：YI-FGSM、YPGD、YMI-FGSM攻击成功，3/5<br>10组攻击全部满足RGB扰动上限 | 生成10张对抗图像<br>生成`step04.json`记录逐攻击结果 |  |  | 主复现路径经RGB裁剪后U/V发生变化<br>严格固定U/V留在4补充对比路径 |
| **4补充** | **对比路径** |  | `cal_y_grad_cmp()` | 对比：只计算Y通道梯度 | 与主复现路径对比 |  | 历史测试：3/3通过<br>`0001.png`、5种攻击、2个ε、两条路径，共20组 |  | 算法或实现对比 | 主路径/固定U/V路径攻击成功6/10和8/10<br>平均PSNR 40.1897/40.8286 dB<br>平均SSIM 0.966538/0.970413 | 本结论仅来自1张图片，不外推为全数据集结论 |
|  |  |  | `project_y_update_cmp()` | 对比：限制RGB扰动和像素范围 |  |  | 两条路径各10/10满足RGB扰动上限 |  | 算法或实现对比 | 固定U/V路径投影后仍满足ε约束 |  |
|  |  |  | `build_attack_result_cmp()` | 对比：生成RGB结果和Y扰动 |  |  | 主路径U/V最大变化0.0078431<br>对比路径算法内部U/V最大变化0 |  | 算法或实现对比 | 严格固定U/V目标实现 | 对比路径输出RGB再次转为YUV后最大变化约2.13e-5，远小于主路径但不再严格为0 |
|  |  |  | `yfgsm_attack_cmp()` | 对比：严格固定U/V的单步攻击 |  |  | 两档ε均未成功：主路径0/2，对比路径0/2 |  | 算法或实现对比 |  |  |
|  |  |  | `yifgsm_attack_cmp()` | 对比：严格固定U/V的迭代攻击 |  |  | 主路径2/2，对比路径2/2 |  | 算法或实现对比 |  |  |
|  |  |  | `ypgd_attack_cmp()` | 对比：严格固定U/V的随机攻击 |  |  | 主路径2/2，对比路径2/2 |  | 算法或实现对比 |  |  |
|  |  |  | `ymifgsm_attack_cmp()` | 对比：严格固定U/V的动量攻击 |  |  | 主路径2/2，对比路径2/2 |  | 算法或实现对比 |  |  |
|  |  |  | `ynifgsm_attack_cmp()` | 对比：严格固定U/V的前瞻攻击 |  |  | 主路径0/2，对比路径2/2 |  | 算法或实现对比 | 本图上成功数提升来自YNI-FGSM两档ε | 固定U/V路径在本图同时获得更高攻击成功数和略高平均PSNR/SSIM |
| 5 | 模型集成 | `s05_imgn_ensemble.py` | `EqualWeightLogitsEnsemble.__init__()` | 保存三个白盒模型 |  | 公式1 |  |  |  |  |  |
|  |  |  | `EqualWeightLogitsEnsemble.forward()` | 计算等权logits |  |  |  |  |  |  |  |
|  |  |  | `cal_ensemble_cross_entropy_loss()` | 公式16 |  |  |  |  |  |  |  |
|  |  | `step05.py` | `main()` | 用一张真实图片验证集成结果 |  |  | `0001.png`，标签321<br>DenseNet161、InceptionV3、GoogLeNet均预测为321<br>三个模型各取1/3权重，集成预测为321<br>集成logits与算术平均最大差值为0 | `step05.json`：保存模型、权重、各模型预测、集成预测、集成损失及等权验证结果 |  |  | 等权logits集成与公式1一致 |
| 6 | CAM区域掩膜 | `s06_cam_mask.py` | `cal_gradcam_plus_plus_low_resolution_map()` | 计算低分辨率Grad-CAM++ |  | 公式7–12、Figure 2 |  |  |  |  |  |
|  |  |  | `cal_gradcam_plus_plus_low_resolution_map.save_activation()` | 保存目标层输出 |  |  |  |  |  |  |  |
|  |  |  | `cal_gradcam_plus_plus_low_resolution_map.save_input_gradient()` | 保存作者使用的输入侧梯度 |  |  |  |  |  |  |  |
|  |  |  | `normalize_cam_map()` | 归一化到0至1 |  |  |  |  |  |  |  |
|  |  |  | `resize_cam_map()` | 复现：使用OpenCV双线性插值 |  |  |  |  |  |  |  |
|  |  |  | `generate_gradcam_plus_plus_map()` | 生成指定插值的热力图 |  |  |  |  |  |  |  |
|  |  |  | `cam_mask()` | 按阈值生成二值掩膜 |  |  |  |  |  |  |  |
|  |  |  | `apply_cam_mask_to_y_perturbation()` | 保留掩膜内的Y扰动 |  |  |  |  |  |  |  |
|  |  | `step06.py` | `save_map_image()` | 保存二维图 |  |  |  |  |  |  |  |
|  |  |  | `save_perturbation_image()` | 保存Y扰动图 |  |  |  |  |  |  |  |
|  |  |  | `main()` | 用一张真实图片验证CAM掩膜 |  |  | `0001.png`，标签及目标类别321<br>阈值0.5，掩膜覆盖19703/89401像素，占22.04%<br>掩膜外Y扰动最大值为0 | `outputs/reproduction/step06`：保存CAM热力图、二值掩膜及掩膜前后Y扰动图<br>`step06.json`：保存掩膜范围、扰动与对比统计 |  |  | Grad-CAM++区域掩膜成功只保留显著区域内的Y扰动 |
| **6补充** | **对比路径** |  | `resize_cam_map_cmp()` | 对比：使用PyTorch双线性插值 | 与主复现路径对比 |  | 历史测试：3/3通过<br>`0001.png`、ResNet50、阈值0.5<br>CAM最大差值4.17e-7，二值掩膜差异0像素<br>两种掩膜均为19703/89401像素 |  | 算法或实现对比 | OpenCV与PyTorch插值只产生浮点微差，没有改变本图阈值掩膜 | 主复现继续使用作者OpenCV插值路径 |
| 7 | 算术编码与PEE | `s0701_arithmetic.py` | `serialize_integer_perturbation_sequence()` | 按行序列化整数扰动 |  | 公式2–6、第9–10行 |  |  |  |  |  |
|  |  |  | `restore_integer_perturbation_shape()` | 恢复扰动数组形状 |  |  |  |  |  |  |  |
|  |  |  | `cal_integer_symbol_frequencies()` | 统计有序频率表 |  |  |  |  |  |  |  |
|  |  |  | `_cal_cumulative_ranges()` | 计算累计频率区间 |  |  |  |  |  |  |  |
|  |  |  | `arithmetic_encode()` | 复现作者仅返回算术码的路径 |  |  | 89401个整数编码为2806–2807 bit算术码<br>6次独立进程结果为2806、2806、2806、2807、2807、2806 bit |  |  |  | 作者仓库含解码函数，但主程序未保存频率表和PEE阈值，也未形成从最终RAE恢复原图的闭环<br>无序`set`使比特流和码长跨进程不完全确定 |
|  |  |  | `_unsigned_integer_to_bits()` | 定宽无符号整数转比特 |  |  |  |  |  |  |  |
|  |  |  | `_signed_integer_to_bits()` | 定宽有符号整数转比特 |  |  |  |  |  |  |  |
|  |  |  | `_bits_to_unsigned_integer()` | 比特转无符号整数 |  |  |  |  |  |  |  |
|  |  |  | `_bits_to_signed_integer()` | 比特转有符号整数 |  |  |  |  |  |  |  |
|  |  | `s0702_pee.py` | `_val_integer_channel()` | 检查二维整数载体 |  |  |  |  |  |  |  |
|  |  |  | `_val_bits()` | 检查二进制序列 |  |  |  |  |  |  |  |
|  |  |  | `_integer_to_bits()` | 整数转定宽比特 |  |  |  |  |  |  |  |
|  |  |  | `_bits_to_integer()` | 定宽比特转整数 |  |  |  |  |  |  |  |
|  |  |  | `generate_checkerboard_prediction()` | 使用四邻域预测半数像素 |  |  |  |  |  |  |  |
|  |  |  | `embed_bits_in_prediction_errors()` | 按公式3、4扩展误差 |  |  |  |  |  |  |  |
|  |  |  | `extract_bits_from_prediction_errors()` | 按公式5、6提取并恢复 |  |  |  |  |  |  |  |
|  |  |  | `_embed_two_checkerboard_passes()` | 先交叉像素再点像素嵌入 |  |  |  |  |  |  |  |
|  |  |  | `select_pee_threshold()` | 按作者1、6、11顺序选择阈值 |  |  |  |  |  |  |  |
|  |  |  | `embed_bits_pee()` | 添加长度头并执行两遍PEE |  |  |  |  |  |  |  |
|  |  |  | `extract_bits_pee()` | 逆序提取两遍PEE |  |  |  |  |  |  |  |
|  |  |  | `embed_payload_uv()` | 将载荷嵌入U和V |  |  |  |  |  |  |  |
|  |  |  | `extract_payload_uv()` | 提取载荷并恢复U和V |  |  |  |  |  |  |  |
|  |  |  | `embed_payload_uv_rounds()` | 按作者方式分轮嵌入长载荷 |  |  | 2500 bit长载荷分2轮完成嵌入 |  |  |  |  |
|  |  | `step07.py` | `generate_sparse_integer_perturbation()` | 生成人工稀疏扰动 |  |  |  |  |  |  |  |
|  |  |  | `main()` | 用人工数组和真实U/V验证闭环 |  |  | `0001.png`，3207 bit载荷<br>阈值1，可用容量83082 bit<br>载荷、U、V及整数扰动全部精确恢复<br>嵌入后取值范围40–204 | `outputs/reproduction/step07`：保存`stego_u.png`和`stego_v.png`<br>`step07.json`：保存算术编码、PEE容量及精确恢复结果 |  |  | PEE主路径完成可逆嵌入与提取 |
| **7补充** | **工程补充** |  | `arithmetic_encode_sup()` | 按作者区间更新方式编码 | 补充作者未提供的工程能力 |  | 89401个整数编码为2807 bit<br>加入频率表后载荷为3207 bit |  | 工程补充 |  |  |
|  |  |  | `arithmetic_decode_sup()` | 使用频率表恢复整数序列 |  |  | 解码后89401个整数全部精确恢复 |  | 工程补充 |  |  |
|  |  |  | `build_arithmetic_payload_sup()` | 将频率表和编码合成载荷 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `parse_arithmetic_payload_sup()` | 从载荷读取频率表和编码 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `parse_arithmetic_payload_sup.read()` | 顺序读取定宽字段 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `extract_payload_uv_rounds_sup()` | 逆序恢复作者多轮PEE |  |  | 2轮提取后载荷、U、V全部精确恢复 |  | 工程补充 |  | 补足作者多轮嵌入的逆过程 |
| 8 | RAE生成与恢复 | `s08_rae.py` | `generate_rae()` | 使用作者短载荷生成YUV载体 |  | Algorithm 2 |  |  |  |  |  |
|  |  |  | `yuv_to_rgb_uint8()` | 生成论文RGB路径 |  |  |  |  |  |  |  |
|  |  | `step08.py` | `parse_arguments()` | 读取生成或独立恢复参数 |  |  |  |  |  |  |  |
|  |  |  | `main()` | 生成RAE，或从自包含PNG执行独立恢复 |  |  | `0001.png`，标签321<br>YMI-FGSM、3次迭代、ε=2/255<br>RAE预测为325，攻击成功<br>作者短载荷120973 bit，阈值6，容量170370 bit | `outputs/reproduction/step08/rae.png`：保存普通RGB格式RAE<br>`step08.json`：保存主路径、恢复对比和工程恢复结果 |  |  | 成功生成仍具攻击性的RAE |
| **8补充** | **对比路径** |  | `recover_rgb_cmp()` | 从论文RGB路径尝试恢复 | 与主复现路径对比 |  | 历史测试：2/2通过<br>`0001.png`普通RGB恢复失败，自包含PNG精确恢复 | **普通RGB恢复路径已放弃** | 算法或实现对比 | 普通RGB往返后Y/U/V分别改变1012/1078/988个值，最大误差4/5/6，载荷无法提取<br>自包含PNG为516555字节，RGB和Y扰动均精确恢复，最大误差0 | 普通RGB文件不能支撑当前实现的无损恢复<br>补充保存精确YUV状态后才能完成可逆闭环 |
| **8补充** | **工程补充** |  | `encode_y_payload_sup()` | 压缩整数Y扰动 | 补充作者未提供的工程能力 |  | 自包含PNG为515837字节，可见像素与主RAE一致<br>NPZ容器为320396字节<br>两条路径均精确恢复原始RGB，最大通道误差0 | `outputs/sup/step08`：保存自包含RAE、两张恢复图和NPZ恢复容器 | 工程补充 |  | 补充状态后可实现逐像素可逆恢复 |
|  |  |  | `recover_saved_rae_sup()` | 仅凭自包含PNG独立恢复并可选核对原图 |  |  | 从磁盘重新读取299×299自包含PNG<br>恢复19185个非零Y扰动位置<br>恢复RGB与原图逐像素一致，最大通道误差0 | `recovered_from_file.png`：独立恢复图<br>`step08_recovery.json`：记录独立恢复结果 | 工程补充 |  | 排除对生成阶段内存变量的依赖 |
|  |  |  | `decode_y_payload_sup()` | 解码并恢复Y扰动形状 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `generate_rae_sup()` | 按Algorithm 2生成YUV载体 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `recover_yuv_sup()` | 提取扰动并恢复原YUV |  |  |  |  | 工程补充 |  |  |
|  |  |  | `recover_yuv_payload_sup()` | 解码作者短载荷 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `cal_rgb_residual_sup()` | 保存颜色转换残差 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `recover_rgb_exact_sup()` | 使用残差精确恢复RGB |  |  |  |  | 工程补充 |  |  |
|  |  |  | `save_recovery_npz_sup()` | 保存精确恢复所需内容 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `load_recovery_npz_sup()` | 读取精确恢复容器 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `save_recovery_png_sup()` | 保存像素不变的自包含恢复PNG |  |  |  |  | 工程补充 |  |  |
|  |  |  | `load_recovery_png_sup()` | 读取自包含恢复PNG |  |  |  |  | 工程补充 |  |  |
|  |  |  | `_load_arithmetic_metadata_sup()` | 读取作者短载荷解码信息 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `recover_png_sup()` | 使用补充状态恢复原始RGB |  |  |  |  | 工程补充 |  |  |
|  |  |  | `generate_rae_rounds_sup()` | 使用多轮PEE生成大载荷RAE |  |  |  |  | 工程补充 |  | 解决单轮载荷上限 |
|  |  |  | `recover_yuv_rounds_sup()` | 逆序提取多轮PEE并恢复原YUV |  |  |  |  | 工程补充 |  |  |
|  |  |  | `save_recovery_png_rounds_sup()` | 保存多轮PEE自包含PNG |  |  |  |  | 工程补充 |  |  |
|  |  |  | `load_recovery_png_rounds_sup()` | 读取多轮PEE自包含PNG |  |  |  |  | 工程补充 |  |  |
|  |  |  | `recover_png_rounds_sup()` | 从多轮PEE自包含PNG恢复原始RGB |  |  |  |  | 工程补充 |  |  |
| 9 | 评价指标 | `s09_metrics.py` | `_val_rgb_pair()` | 检查同形状RGB图片 |  | Table 2–5 |  |  |  |  |  |
|  |  |  | `cal_psnr()` | 计算PSNR |  |  |  |  |  |  |  |
|  |  |  | `cal_ssim()` | 计算SSIM |  |  |  |  |  |  |  |
|  |  |  | `cal_mean_ciede2000()` | 计算平均CIEDE2000 |  |  |  |  |  |  |  |
|  |  |  | `cal_recovery_metrics()` | 统计逐像素恢复结果 |  |  |  |  |  |  |  |
|  |  |  | `cal_attack_success_rate()` | 计算论文或严格ASR |  |  |  |  |  |  |  |
|  |  |  | `cal_white_box_attack_success_rate()` | 计算白盒ASR |  |  |  |  |  |  |  |
|  |  |  | `cal_black_box_attack_success_rate()` | 计算黑盒ASR |  |  |  |  |  |  |  |
|  |  | `step09.py` | `main()` | 计算一张真实RAE的全部指标 |  |  | `0001.png`，样本数1<br>PSNR 36.4612 dB，SSIM 0.9671，平均CIEDE2000 2.2403<br>白盒ASR 2/3=66.67%，黑盒ASR 1/2=50%<br>恢复RGB与原图逐像素相同，最大误差0 | `step09.json`：保存视觉质量、5个模型预测、白盒与黑盒ASR及恢复指标 |  |  | 完成单张真实RAE的论文指标计算 |
| 10 | 基线和消融实验 | `s1002_ablation.py` | `generate_method_adversarial_rgb()` | 按RDH、YUV、YACK、ENS或OURS生成攻击图 |  | Tables 2–5 | YUV完成1000张×10组实验<br>RDH只完成入口与恢复闭环验证<br>YUV使用RGB攻击，YACK使用Y通道攻击 | YUV数据集实验完成<br>**RDH全量未复现并已明确放弃** |  |  | 论文已有基线不归入额外`_cmp`实验 |
|  |  |  | `cal_integer_y_perturbation()` | 计算取整后的Y通道差值 |  |  |  |  |  |  |  |
|  |  |  | `apply_binary_mask_to_integer_y_perturbation()` | 保留CAM区域内的整数扰动 |  |  |  |  |  |  |  |
|  |  | `s1005_rdh.py` | `select_rdh_superpixel_shape()` | 为攻击和扰动范围选择超像素尺寸 |  | Tables 2–4 | FGSM为1×1<br>迭代攻击2/255为1×2、4/255为1×3 | 参数入口已验证 |  |  | 2/255参数属于显式复现假设 |
|  |  |  | `smooth_rgb_perturbation_by_superpixels()` | 对块内RGB扰动取整平均 |  |  | 块内扰动一致性测试通过 |  |  |  | 复现RDH后平滑结构 |
|  |  |  | `_cal_superpixel_payload()` | 生成块扰动和裁剪残差 |  |  |  |  |  |  | 支持边界像素精确恢复 |
|  |  |  | `_expand_superpixel_deltas()` | 将块扰动还原到整幅图像 |  |  |  |  |  |  |  |
|  |  |  | `_stack_rgb_channels()` | 将RGB三通道堆叠为二维载体 |  |  |  |  |  |  |  |
|  |  |  | `_unstack_rgb_channels()` | 将二维载体恢复为RGB |  |  |  |  |  |  |  |
|  |  |  | `prepare_rdh_candidate_pro()` | 压缩超像素扰动并嵌入整幅RGB |  |  | 真实`0001.png`人工+2扰动可嵌入<br>149798 bit，阈值6 | CPU真实图预检通过 | 工程补充 | 使用PEE替代未公开RHM | 后端字段固定为`PEE_reference_substitute` |
|  |  |  | `finalize_rdh_candidate_pro()` | 提取扰动并逐像素恢复原图 |  |  | 2项RDH测试通过<br>10张测试运行6/10组后按用户要求停止<br>成功生成28张次，28/28精确恢复，最大通道误差0<br>实际自适应超像素为1×2至1×8 | 已证明参考RDH入口可运行并形成恢复闭环 | 工程补充 | 使用`PEE_reference_substitute` | **RDH全量复现已放弃，不再运行剩余配置** |
|  |  | `step10.py` | `parse_arguments()` | 读取实验参数 |  |  |  |  |  |  |  |
|  |  |  | `load_experiment_images()` | 读取实验图片 |  |  |  |  |  |  |  |
|  |  |  | `generate_cam_masks()` | 生成统一作者CAM掩膜 |  |  |  |  |  |  |  |
|  |  |  | `load_attack_model()` | 加载默认模型、集成模型或指定单源模型 |  | Tables 3–5 | 支持InceptionV3、GoogLeNet、DenseNet161单源模型<br>2项相关测试通过 | 尚未运行Tables 3–4数据集实验 |  |  | 显式单源模型用于分别生成并评价对应白盒RAE |
|  |  |  | `create_reversible_example()` | 按作者循环生成单张RAE |  |  |  |  |  |  |  |
|  |  |  | `predict_rgb_arrays()` | 分批预测RGB数组 |  |  |  |  |  |  |  |
|  |  |  | `load_clean_prediction_rows()` | 读取第3步干净预测 |  |  |  |  |  |  |  |
|  |  |  | `cal_checkpoint_rows()` | 汇总1、10和最终样本结果 |  |  |  |  |  |  |  |
|  |  |  | `write_summary_csv()` | 保存汇总表 |  |  |  |  |  |  |  |
|  |  |  | `load_existing_results()` | 读取断点结果 |  |  |  |  |  |  |  |
|  |  |  | `main()` | 运行选定消融实验 |  |  | `0001.png`，YACK、ENS、OURS三种方法<br>5种攻击、2个ε，共30组配置，30/30生成成功<br>PSNR范围32.6903–41.9863 dB<br>SSIM范围0.9367–0.9857<br>平均CIEDE2000范围1.2161–2.4700 | `outputs/reproduction/step10`：保存30张RAE<br>`step10_1_images.csv`：保存30组配置对5个模型的150行汇总<br>`step10_1_images.json`：保存30组逐样本明细 |  |  | 已完成1张图全配置流程<br>不能作为论文1000张Table 5最终结果 |
| **10补充** | **Pro工程加速** | `s1003_pro.py` | `prepare_rae_candidate_pro()` | CPU进程并行生成单张RAE候选 | 保留串行版并加速大样本实验 | Table 5 |  |  | 工程补充 |  | 工程优势：并行处理颜色转换、PEE与算术编码 |
|  |  |  | `finalize_rae_candidate_pro()` | CPU进程并行计算恢复和图像指标 |  |  |  |  | 工程补充 |  |  |
|  |  | `step10_pro.py` | `parse_arguments_pro()` | 读取GPU批次、CPU进程及单源模型参数 |  |  | OURS使用batch 2/workers 4<br>YACK使用1/2<br>ENS、YUV使用3/6<br>结果后缀成功隔离四组文件 | 参数已用于四种方法1000张正式实验 | 工程补充 | 不同批次使`0280.png`的ResNet50 CAM类别出现68/66差异 | 防止覆盖结果，同时必须在跨方法比较中备注并行参数差异 |
|  |  |  | `load_image_pro()` | 多线程读取单张图片 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `load_experiment_images_pro()` | 多线程读取实验图片 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `generate_cam_masks_pro()` | GPU批量生成CAM掩膜并自动降低OOM批次 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `create_reversible_batch_pro()` | GPU批量攻击并调度CPU进程生成RAE |  |  |  |  | 工程补充 |  |  |
|  |  |  | `create_reversible_examples_pro()` | 分批生成全部RAE并自动降低OOM批次 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `save_artifacts_pro()` | 保存Pro版独立产物 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `main()` | 运行GPU批处理与CPU并行实验 |  | Table 5 | OURS：9382/10000，856–988/组，6.30h<br>YACK：9672/10000，886–998/组，4.43h<br>ENS：8668/10000，800–921/组，4.84h<br>YUV：9084/10000，816–967/组，1.94h<br>四组CSV与JSON的配置数、样本覆盖、生成数、视觉指标和恢复率交叉核对一致 | OURS、YACK、ENS、YUV四种方法的1000张×10组数据已生成<br>**RDH全量未复现并已放弃** | 工程补充 | 成功样本内存恢复均为100%<br>YACK、ENS、YUV未保存全量PNG，不能表述为文件恢复100%<br>ASR和视觉指标以`generated_count`为分母<br>少数生成阶段Inception预测与最终批量评价相差1张，Table统一采用CSV最终评价 | 工程优势：结果文件隔离并支持断点续跑<br>实验结果受不同batch/workers影响，不能把耗时差异表述为算法速度优势 |
| **10补充** | **容量失败多轮PEE** | `s1004_capacity_recovery.py` | `prepare_rae_candidate_rounds_sup()` | 使用多轮PEE生成RAE候选 | 只处理主路径容量失败样本 | Table 5 |  |  | 工程补充 |  | 不替换原978张主路径结果 |
|  |  |  | `finalize_rae_candidate_rounds_sup()` | 恢复多轮PEE候选并计算指标 |  |  |  |  | 工程补充 |  |  |
|  |  | `step10_capacity_sup.py` | `parse_arguments_capacity_sup()` | 读取容量补充参数 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `load_failed_image_names_sup()` | 从1000张JSON读取失败图片名 |  |  |  |  | 工程补充 |  | 只得到原22张失败图 |
|  |  |  | `load_failed_images_sup()` | 读取失败图片及标签 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `save_capacity_artifacts_sup()` | 保存多轮PEE产物并从磁盘恢复验证 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `cal_capacity_summary_sup()` | 汇总补充测试和五模型结果 |  |  |  |  | 工程补充 |  |  |
|  |  |  | `write_capacity_csv_sup()` | 保存22张逐图CSV |  |  |  |  | 工程补充 |  |  |
|  |  |  | `main()` | 仅重跑单轮PEE失败样本 |  |  | 22张全部生成成功并精确恢复，最大通道误差0<br>20张使用2轮PEE，2张重跑后1轮即可；平均1.909轮，最大2轮<br>耗时268.92s；平均PSNR 32.7603 dB，SSIM 0.9264，CIEDE2000 2.9629<br>与原978张合并后恢复1000/1000<br>合并平均PSNR 41.0573 dB，SSIM 0.9771，CIEDE2000 1.4368<br>合并五模型成功数为1000、735、784、421、487 | `step10_capacity_sup_22_images.csv`：22张逐图结果<br>`step10_capacity_sup_22_images.json`：汇总和逐图明细<br>`outputs/reproduction/step10_capacity_sup`与`outputs/sup/step10_capacity_sup`各22张PNG | 工程补充 |  | 多轮PEE补齐22张容量失败样本<br>高载荷样本的视觉质量低于原978张 |
