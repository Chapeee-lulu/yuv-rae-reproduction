# ImageNet图片预处理与模型推理

import csv
import re
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import functional as transform_function

IMAGE_SIZE = (299, 299)
IMGN_MEAN = (0.485, 0.456, 0.406)
IMGN_STANDARD_DEVIATION = (0.229, 0.224, 0.225)
MODEL_NAMES = ("inception_v3", "googlenet", "densenet161", "resnet50", "vgg19")
WHITE_BOX_MODEL_NAMES = ("inception_v3", "googlenet", "densenet161")
BLACK_BOX_MODEL_NAMES = ("resnet50", "vgg19")


def load_imgn_val_labels(label_file: Path) -> dict[str, int]:  # 读取外部验证集标签
    if not label_file.is_file():
        raise FileNotFoundError(f"验证标签文件不存在：{label_file}")

    labels: dict[str, int] = {}
    with label_file.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != ["id", "true_label", "author_file"]:
            raise ValueError("验证标签表字段必须为id、true_label、author_file")
        for row in reader:
            image_id = row["id"]
            if re.fullmatch(r"\d{4}", image_id) is None:
                raise ValueError(f"图片编号无效：{image_id}")
            filename = f"{image_id}.png"
            if filename in labels:
                raise ValueError(f"图片编号重复：{image_id}")
            label = int(row["true_label"])
            if not 0 <= label <= 999:
                raise ValueError(f"ImageNet验证标签必须在0至999之间：{label}")
            labels[filename] = label
    return labels


def generate_imgn_labels(  # 根据三个白盒模型一致预测生成标签
    model_predictions: dict[str, list[int]],
    model_names: tuple[str, ...] = WHITE_BOX_MODEL_NAMES,
) -> list[int | None]:
    if not model_names:
        raise ValueError("至少需要一个标签生成模型")
    missing_models = [name for name in model_names if name not in model_predictions]
    if missing_models:
        raise KeyError(f"缺少模型预测：{missing_models}")
    prediction_lengths = {len(model_predictions[name]) for name in model_names}
    if len(prediction_lengths) != 1:
        raise ValueError("用于生成标签的模型预测数量不一致")

    generated_labels: list[int | None] = []
    for index in range(prediction_lengths.pop()):
        predictions = [int(model_predictions[name][index]) for name in model_names]
        if any(not 0 <= prediction <= 999 for prediction in predictions):
            raise ValueError("模型预测标签必须在0至999之间")
        generated_labels.append(
            predictions[0] if len(set(predictions)) == 1 else None
        )
    return generated_labels


def load_generated_imgn_labels(  # 读取Step 03生成且验证通过的标签
    label_file: Path,
) -> dict[str, int]:
    if not label_file.is_file():
        raise FileNotFoundError(
            f"Step 03生成标签不存在，请先运行script/step03.py：{label_file}"
        )

    labels: dict[str, int] = {}
    with label_file.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_fields = [
            "filename",
            "generated_label",
            "val_label",
            "white_box_consensus",
            "matches_val",
        ]
        if reader.fieldnames != required_fields:
            raise ValueError(f"Step 03标签表字段必须为{required_fields}")
        for row in reader:
            filename = row["filename"]
            if re.fullmatch(r"\d{4}\.png", filename) is None:
                raise ValueError(f"生成标签中的图片名无效：{filename}")
            if row["white_box_consensus"] != "True" or row["matches_val"] != "True":
                continue
            label = int(row["generated_label"])
            if not 0 <= label <= 999:
                raise ValueError(f"生成标签必须在0至999之间：{label}")
            labels[filename] = label
    return labels


def convert_image_to_unit_tensor(image: Image.Image) -> torch.Tensor:  # 转为0到1张量
    rgb_image = image.convert("RGB")
    return transform_function.pil_to_tensor(rgb_image).to(torch.float32) / 255.0


def imgn_input(image: Image.Image) -> torch.Tensor:  # 生成非归一化输入[默认直接返回]
    return convert_image_to_unit_tensor(image)


def add_batch_dimension(image_tensor: torch.Tensor) -> torch.Tensor:  # 增加批次维度
    return image_tensor.unsqueeze(0)


class ImgnPngDataset(Dataset):
    def __init__(  # 保存不含标签的原始图片路径
        self,
        image_paths: list[Path],
        normalize_input: bool = False,
    ):
        self.image_paths = image_paths
        self.normalize_input = normalize_input

    def __len__(self) -> int:  # 返回图片数量
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:  # 读取无标签图片
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            input_tensor = imgn_input(image)
        if self.normalize_input:
            input_tensor = imgn_input_cmp(input_tensor)
        return input_tensor, image_path.name


def load_imgn_model(model_name: str, device: torch.device) -> nn.Module:  # 加载预训练模型
    model_builders = {
        "inception_v3": (models.inception_v3, models.Inception_V3_Weights.IMAGENET1K_V1),
        "googlenet": (models.googlenet, models.GoogLeNet_Weights.IMAGENET1K_V1),
        "densenet161": (models.densenet161, models.DenseNet161_Weights.IMAGENET1K_V1),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
        "vgg19": (models.vgg19, models.VGG19_Weights.IMAGENET1K_V1),
    }
    if model_name not in model_builders:
        raise ValueError(f"未知模型：{model_name}")

    builder, weights = model_builders[model_name]
    model = builder(weights=weights)
    model.eval()
    return model.to(device)


@torch.inference_mode()
def predict_class_indices(model: nn.Module, input_batch: torch.Tensor) -> torch.Tensor:  # 预测类别编号，取logits的argmax
    logits = model(input_batch)
    return logits.argmax(dim=1)


@torch.inference_mode()
def predict_imgn_dataset(  # 预测数据集
    model: nn.Module,
    dataset: ImgnPngDataset,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[list[int], list[str]]:
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: list[int] = []
    filenames: list[str] = []

    for input_batch, filename_batch in data_loader:
        prediction_batch = predict_class_indices(model, input_batch.to(device))
        predictions.extend(prediction_batch.cpu().tolist())
        filenames.extend(filename_batch)

    return predictions, filenames


def imgn_input_cmp(unit_tensor: torch.Tensor) -> torch.Tensor:  # 生成归一化输入[对比]
    return transform_function.normalize(
        unit_tensor,
        mean=IMGN_MEAN,
        std=IMGN_STANDARD_DEVIATION,
    )
