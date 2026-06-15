"""SegFormer-B0 with Boundary Enhancement Head.

The boundary head takes the fused decoder feature (after linear_fuse + BN + ReLU)
and predicts a binary edge map, supervised by boundary labels generated
from semantic ground truth via morphological operations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class BoundaryHead(nn.Module):
    """Lightweight boundary prediction head: Conv3x3 -> BN -> ReLU -> Conv1x1."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.conv2(x)


class SegFormerBoundary(nn.Module):
    """SegFormer-B0 with semantic head + boundary enhancement head.

    The fused decoder feature (after linear_fuse, BN, activation, dropout)
    is fed into both the semantic classifier and the boundary head.
    """

    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()

        model_name = "nvidia/mit-b0"
        if pretrained:
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                model_name, num_labels=num_classes, ignore_mismatched_sizes=True
            )
        else:
            config = SegformerForSemanticSegmentation.config_class.from_pretrained(
                model_name, num_labels=num_classes
            )
            self.segformer = SegformerForSemanticSegmentation(config)

        self.decoder_hidden_dim = self.segformer.config.decoder_hidden_size  # 256
        self.boundary_head = BoundaryHead(self.decoder_hidden_dim)

    def _fuse_encoder_features(self, hidden_states):
        """Replicate the SegformerDecodeHead forward up to (but excluding) the classifier.

        This matches the exact logic from transformers' SegformerDecodeHead.forward:
        - Project each stage with SegformerMLP
        - Upsample all to stage-1 spatial size
        - Concat in REVERSE order
        - linear_fuse (Conv1d/Conv2d), batch_norm, activation, dropout

        Returns fused feature of shape (B, decoder_hidden_dim, H/4, W/4).
        """
        decode_head = self.segformer.decode_head
        batch_size = hidden_states[0].shape[0]
        target_size = hidden_states[0].shape[2:]  # (H/4, W/4)

        all_hidden_states = []
        for encoder_hidden_state, mlp in zip(hidden_states, decode_head.linear_c):
            # encoder_hidden_state: (B, C_i, H_i, W_i)
            H_i, W_i = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]

            # mlp forward: flatten(2).transpose(1,2).proj → (B, H_i*W_i, D)
            hidden_state = mlp(encoder_hidden_state)

            # permute(0, 2, 1) → (B, D, H_i*W_i), then reshape → (B, D, H_i, W_i)
            hidden_state = hidden_state.permute(0, 2, 1)
            hidden_state = hidden_state.reshape(batch_size, -1, H_i, W_i)

            # Upsample to target size
            hidden_state = F.interpolate(
                hidden_state, size=target_size, mode="bilinear", align_corners=False
            )
            all_hidden_states.append(hidden_state)

        # Concat in reverse order (matches official implementation)
        fused = torch.cat(all_hidden_states[::-1], dim=1)  # (B, 4*D, H, W)

        # linear_fuse, batch_norm, activation, dropout
        fused = decode_head.linear_fuse(fused)         # Conv2d: (B, D, H, W)
        fused = decode_head.batch_norm(fused)
        fused = decode_head.activation(fused)
        fused = decode_head.dropout(fused)

        return fused

    def forward(self, x):
        """Return (semantic_logits, boundary_logits).

        semantic_logits: (B, num_classes, H/4, W/4)
        boundary_logits: (B, 1, H/4, W/4)
        """
        outputs = self.segformer(x, output_hidden_states=True)
        semantic_logits = outputs.logits
        hidden_states = outputs.hidden_states
        fused = self._fuse_encoder_features(hidden_states)
        boundary_logits = self.boundary_head(fused)
        return semantic_logits, boundary_logits

    def get_fused_feature(self, x):
        """Return the fused decoder feature for analysis."""
        outputs = self.segformer(x, output_hidden_states=True)
        return self._fuse_encoder_features(outputs.hidden_states)
