import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as tv_models

try:
    import torchviz
    _HAS_TORCHVIZ = True
except ImportError:
    _HAS_TORCHVIZ = False


class motionGradeCNN(nn.Module):
    """Custom CNN for motion grade classification with Grad-CAM support."""

    def __init__(self, num_classes=4):
        super().__init__()
        assert 1 <= num_classes <= 5

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.GroupNorm(8, 32)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.GroupNorm(8, 32)
        self.relu2 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.GroupNorm(8, 64)
        self.relu3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.GroupNorm(8, 64)
        self.relu4 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.GroupNorm(8, 128)
        self.relu5 = nn.LeakyReLU()
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.GroupNorm(8, 128)
        self.relu6 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.GroupNorm(16, 256)
        self.relu7 = nn.LeakyReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.GroupNorm(16, 256)
        self.relu8 = nn.LeakyReLU()
        self.pool5 = nn.MaxPool2d(2, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, num_classes)
        self.num_classes = num_classes

        self.conv1.register_full_backward_hook(self.save_gradient)
        self.conv1.register_forward_hook(self.save_activation)
        self.gradient = None
        self.activation = None

    def save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def save_activation(self, module, input, output):
        self.activation = output

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.pool4(x)
        x = self.relu8(self.bn8(self.conv8(x)))
        x = self.pool5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc1(x)

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=1)


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 for 1-channel input, pretrained on ImageNet.
    Stem conv is averaged from 3→1 channel to preserve learned edge detectors.
    """

    def __init__(self, num_classes=4, freeze_blocks=7):
        super().__init__()
        base = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        base.features[0][0] = new_conv

        self.features = base.features
        self.avgpool = base.avgpool
        self.num_classes = num_classes

        in_features = base.classifier[1].in_features
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))

        for i, block in enumerate(self.features):
            if i < freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ConvNeXtSmall(nn.Module):
    """ConvNeXt-Small for 1-channel input, pretrained on ImageNet."""

    def __init__(self, num_classes=4, freeze_stages=6):
        super().__init__()
        base = tv_models.convnext_small(weights=tv_models.ConvNeXt_Small_Weights.IMAGENET1K_V1)

        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        base.features[0][0] = new_conv

        self.features = base.features
        self.avgpool = base.avgpool
        self.num_classes = num_classes

        in_features = base.classifier[2].in_features
        self.classifier = nn.Sequential(
            base.classifier[0],  # LayerNorm2d
            nn.Flatten(1),
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

        for i, block in enumerate(self.features):
            if i < freeze_stages:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


def build_model(num_classes: int, arch: str = 'custom') -> nn.Module:
    if arch == 'custom':
        return motionGradeCNN(num_classes=num_classes)
    elif arch == 'efficientnet':
        return EfficientNetB0(num_classes=num_classes)
    elif arch == 'convnext':
        return ConvNeXtSmall(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch!r}. Choose 'custom', 'efficientnet', or 'convnext'.")


def set_finetune_mode(model: nn.Module, arch: str, phase: str) -> None:
    """
    Configure which layers are trainable for fine-tuning.

    phase='head'  — freeze backbone, train only classifier
    phase='full'  — unfreeze all parameters
    """
    if phase == 'head':
        for param in model.parameters():
            param.requires_grad = False
        if arch in ('efficientnet', 'convnext'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        else:
            for param in model.fc1.parameters():
                param.requires_grad = True
    elif phase == 'full':
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown fine-tune phase: {phase!r}")
