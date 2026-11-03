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


def _adapt_stem(old_conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Adapt a pretrained RGB stem conv to `in_channels` input channels.

    in_channels == 3 keeps the pretrained weights untouched (e.g. a 3-slice
    z-context stack lines up exactly with the pretrained 3-channel stem).
    Otherwise the weights are averaged to 1 channel and repeated (scaled by
    1/in_channels to keep output magnitude comparable) across in_channels —
    for in_channels=1 this reduces to the original averaging behavior.
    """
    new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride, padding=old_conv.padding, bias=False)
    if in_channels == old_conv.in_channels:
        new_conv.weight.data = old_conv.weight.data.clone()
    else:
        avg = old_conv.weight.data.mean(dim=1, keepdim=True)
        new_conv.weight.data = avg.repeat(1, in_channels, 1, 1) / in_channels
    return new_conv


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 for grayscale (or z-context-stacked) input, pretrained on ImageNet.
    Stem conv is adapted from 3 channels to `in_channels` — see `_adapt_stem`.
    """

    def __init__(self, num_classes=4, freeze_blocks=7, in_channels=1):
        super().__init__()
        base = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        base.features[0][0] = _adapt_stem(base.features[0][0], in_channels)

        self.features = base.features
        self.avgpool = base.avgpool
        self.num_classes = num_classes
        self.in_features = base.classifier[1].in_features
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.in_features, num_classes))

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
    """ConvNeXt-Small for grayscale (or z-context-stacked) input, pretrained on ImageNet."""

    def __init__(self, num_classes=4, freeze_stages=6, in_channels=1):
        super().__init__()
        base = tv_models.convnext_small(weights=tv_models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        base.features[0][0] = _adapt_stem(base.features[0][0], in_channels)

        self.features = base.features
        self.avgpool = base.avgpool
        self.num_classes = num_classes
        self.in_features = base.classifier[2].in_features
        self.classifier = nn.Sequential(
            base.classifier[0],  # LayerNorm2d
            nn.Flatten(1),
            nn.Dropout(0.3),
            nn.Linear(self.in_features, num_classes),
        )

        for i, block in enumerate(self.features):
            if i < freeze_stages:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


class AttentionMIL(nn.Module):
    """
    Attention-based multiple-instance learning head (Ilse et al., 2018) over a
    bag of per-scan slice instances, in place of averaging independent
    per-slice predictions after the fact. Each instance is embedded through a
    shared EfficientNet/ConvNeXt backbone, then combined into one scan-level
    prediction via learned attention weights, so the model can emphasize the
    slices that actually carry the motion artifact instead of diluting a
    localized artifact across a mean over the whole volume.
    """

    def __init__(self, num_classes=4, backbone_arch='efficientnet', in_channels=1,
                freeze_blocks=7, attn_hidden=128, chunk_size=16):
        super().__init__()
        if backbone_arch == 'efficientnet':
            backbone = EfficientNetB0(num_classes=num_classes, freeze_blocks=freeze_blocks, in_channels=in_channels)
        elif backbone_arch == 'convnext':
            backbone = ConvNeXtSmall(num_classes=num_classes, freeze_stages=freeze_blocks, in_channels=in_channels)
        else:
            raise ValueError(f"AttentionMIL only supports 'efficientnet' or 'convnext', got {backbone_arch!r}")

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.in_features
        self.num_classes = num_classes
        self.chunk_size = chunk_size

        self.attention = nn.Sequential(
            nn.Linear(in_features, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, H, W) instances -> (N, in_features) embeddings, processed
        in chunks to bound peak forward-pass memory for large bags."""
        chunks = []
        for start in range(0, x.size(0), self.chunk_size):
            h = self.features(x[start:start + self.chunk_size])
            h = self.avgpool(h)
            chunks.append(torch.flatten(h, 1))
        return torch.cat(chunks, dim=0)

    def forward(self, x: torch.Tensor):
        """x: (N, C, H, W) — one bag (all instances of one scan).
        Returns (logits: (1, num_classes), attn_weights: (N,))."""
        h = self.embed(x)
        a = torch.softmax(self.attention(h).squeeze(-1), dim=0)
        bag = (a.unsqueeze(-1) * h).sum(dim=0, keepdim=True)
        return self.classifier(bag), a


def build_model(num_classes: int, arch: str = 'custom', in_channels: int = 1, mil: bool = False) -> nn.Module:
    if mil:
        if arch not in ('efficientnet', 'convnext'):
            raise ValueError(f"MIL mode requires arch 'efficientnet' or 'convnext', got {arch!r}")
        return AttentionMIL(num_classes=num_classes, backbone_arch=arch, in_channels=in_channels)
    if arch == 'custom':
        if in_channels != 1:
            raise ValueError("'custom' arch only supports in_channels=1")
        return motionGradeCNN(num_classes=num_classes)
    elif arch == 'efficientnet':
        return EfficientNetB0(num_classes=num_classes, in_channels=in_channels)
    elif arch == 'convnext':
        return ConvNeXtSmall(num_classes=num_classes, in_channels=in_channels)
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
        if isinstance(model, AttentionMIL):
            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.attention.parameters():
                param.requires_grad = True
        elif arch in ('efficientnet', 'convnext'):
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
