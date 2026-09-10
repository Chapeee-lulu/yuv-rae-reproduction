# ImageNet等权logits集成

import torch
from torch import nn
from torch.nn import functional as torch_functional


ENSEMBLE_MODEL_NAMES = ("densenet161", "inception_v3", "googlenet")


class EqualWeightLogitsEnsemble(nn.Module):
    def __init__(self, models: list[nn.Module]):  # 保存三个白盒模型
        super().__init__()
        if len(models) != 3:
            raise ValueError("必须提供3个模型")
        self.models = nn.ModuleList(models)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:  # 计算等权logits
        logits = [model(input_batch) for model in self.models]
        return logits[0] / 3 + logits[1] / 3 + logits[2] / 3


def cal_ensemble_cross_entropy_loss(  # 公式16
    ensemble_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return torch_functional.cross_entropy(ensemble_logits, labels)
