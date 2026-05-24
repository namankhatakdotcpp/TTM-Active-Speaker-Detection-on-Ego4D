"""
ResNet-18 audio encoder for TTM binary classification.

Input:  [B, 1, N_MELS=64, T=50]  — single-channel log-Mel spectrogram
Output: [B, 2]                    — TTM logits   (for classification)
        [B, 512]                  — embedding     (for fusion with video)

ResNet-18 is adapted for single-channel audio input (conv1 kernel changed
from 3-channel RGB to 1-channel spectrogram).
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class AudioEncoder(nn.Module):
    """
    ResNet-18 adapted for log-Mel spectrogram input.

    Args:
        num_classes:  Output classes (2 for TTM binary).
        pretrained:   Load ImageNet weights (adapted for 1-channel input).
        dropout:      Dropout before classification head.
        embed_dim:    Dimension of the embedding output (for fusion).
    """

    EMBED_DIM = 512     # ResNet-18 penultimate feature size

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # Adapt conv1: RGB (3ch) → spectrogram (1ch)
        # Average the 3-channel pretrained weights across channel dim
        old_conv  = backbone.conv1
        new_conv  = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv

        # Remove the final FC — we replace it with our own head
        in_features   = backbone.fc.in_features           # 512
        backbone.fc   = nn.Identity()

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, N_MELS, T]
        Returns:
            logits: [B, num_classes]
        """
        features = self.backbone(x)    # [B, 512]
        return self.head(features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the 512-d embedding (before classification head).
        Used for audio-video fusion.

        Args:
            x: [B, 1, N_MELS, T]
        Returns:
            embed: [B, 512]
        """
        return self.backbone(x)

    def count_parameters(self) -> dict:
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trained}


# ── quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = AudioEncoder(pretrained=False).to(device)
    x      = torch.randn(4, 1, 64, 50, device=device)   # batch of 4 clips
    logits = model(x)
    embed  = model.encode(x)
    print(f"Input:     {x.shape}")
    print(f"Logits:    {logits.shape}")
    print(f"Embedding: {embed.shape}")
    params = model.count_parameters()
    print(f"Params — total: {params['total']/1e6:.1f}M  trainable: {params['trainable']/1e6:.1f}M")
