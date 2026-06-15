"""DeepLabV3+ baseline using torchvision."""
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


class DeepLabV3PlusWrapper(nn.Module):
    """DeepLabV3+ wrapper that returns a single output tensor."""

    def __init__(self, num_classes=12):
        super().__init__()
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

        # Replace heads for CamVid
        in_channels = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        aux_in = self.model.aux_classifier[-1].in_channels
        self.model.aux_classifier[-1] = nn.Conv2d(aux_in, num_classes, kernel_size=1)

    def forward(self, x):
        result = self.model(x)
        # Return "out" (main output); "aux" is only used during training
        if self.training:
            # Return main output only; aux loss is not used in our pipeline
            return result["out"]
        return result["out"]


def create_deeplabv3(num_classes=12):
    """Create DeepLabV3+ with ResNet-50 backbone for CamVid."""
    return DeepLabV3PlusWrapper(num_classes=num_classes)
