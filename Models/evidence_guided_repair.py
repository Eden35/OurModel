"""Paper-aligned evidence-guided selective repair network.

The legacy :mod:`SAUM_Net` is kept for backwards compatibility.  This module
implements the network described in the supplied Methods section: two
interacting encoders, probabilistic SDF uncertainty, Otsu region selection,
self-excluding reference attention, residual repair, and a gain--harm router.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity, inplace=True)


def make_stage(in_channels: int, out_channels: int, blocks: int, stride: int):
    layers = [BasicBlock(in_channels, out_channels, stride)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class InteractionBlock(nn.Module):
    """Compact recurrent interaction state with channel attention."""

    def __init__(self, structure_channels: int, context_channels: int, state_channels: int = 16):
        super().__init__()
        self.update = ConvNormAct(structure_channels + context_channels + state_channels, state_channels)
        hidden = max(4, state_channels // 4)
        self.attention = nn.Sequential(
            nn.Linear(state_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, state_channels),
            nn.Sigmoid(),
        )
        self.structure_projection = nn.Conv2d(state_channels, structure_channels, 1)
        self.context_projection = nn.Conv2d(state_channels, context_channels, 1)
        self.state_channels = state_channels

    def forward(self, structure, context, previous):
        if previous is None:
            previous = structure.new_zeros(structure.size(0), self.state_channels, structure.size(2), structure.size(3))
        elif previous.shape[-2:] != structure.shape[-2:]:
            previous = F.interpolate(previous, size=structure.shape[-2:], mode="bilinear", align_corners=False)
        state = self.update(torch.cat([structure, context, previous], dim=1)) + previous
        weights = self.attention(state.mean(dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)
        state = state * weights
        return structure + self.structure_projection(state), context + self.context_projection(state), state


class Encoder(nn.Module):
    def __init__(self, channels: Sequence[int] = (32, 64, 128, 256), blocks: Sequence[int] = (3, 4, 6, 3)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stages = nn.ModuleList([
            make_stage(channels[0], channels[0], blocks[0], 1),
            make_stage(channels[0], channels[1], blocks[1], 2),
            make_stage(channels[1], channels[2], blocks[2], 2),
            make_stage(channels[2], channels[3], blocks[3], 2),
        ])

    def forward_stem(self, x):
        return self.stem(x)


class UpDecoder(nn.Module):
    def __init__(self, channels: Sequence[int] = (32, 64, 128, 256), output_channels: int = 32):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.merge3 = BasicBlock(channels[2] * 2, channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.merge2 = BasicBlock(channels[1] * 2, channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.merge1 = BasicBlock(channels[0] * 2, channels[0])
        self.up0 = nn.ConvTranspose2d(channels[0], channels[0], 2, stride=2)
        self.merge0 = BasicBlock(channels[0], channels[0])
        self.output = nn.ConvTranspose2d(channels[0], output_channels, 2, stride=2)

    @staticmethod
    def _resize(x, reference):
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, features, output_size):
        f1, f2, f3, f4 = features
        x = self._resize(self.up3(f4), f3)
        x = self.merge3(torch.cat([x, f3], dim=1))
        x = self._resize(self.up2(x), f2)
        x = self.merge2(torch.cat([x, f2], dim=1))
        x = self._resize(self.up1(x), f1)
        x = self.merge1(torch.cat([x, f1], dim=1))
        x = self.merge0(self.up0(x))
        x = F.relu(self.output(x), inplace=True)
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class DualBranchUncertaintyNet(nn.Module):
    """Dual structure/context base network and uncertainty descriptors."""

    def __init__(self, num_classes: int = 2, state_channels: int = 16):
        super().__init__()
        if num_classes != 2:
            raise ValueError("The paper formulation is binary and expects num_classes=2")
        channels = (32, 64, 128, 256)
        self.structure_encoder = Encoder(channels)
        self.context_encoder = Encoder(channels)
        self.interactions = nn.ModuleList([
            InteractionBlock(c, c, state_channels) for c in channels
        ])
        self.structure_decoder = UpDecoder(channels, output_channels=32)
        self.context_decoder = UpDecoder(channels, output_channels=32)
        self.structure_head = nn.Conv2d(32, 2, 1)
        self.context_head = nn.Conv2d(32, 2, 1)
        self.sdf_head = nn.Conv2d(32, 2, 1)
        self.state_channels = state_channels

    def forward(self, x) -> Dict[str, torch.Tensor]:
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError("Expected an RGB tensor with shape [B,3,H,W]")
        fs = self.structure_encoder.forward_stem(x)
        fc = self.context_encoder.forward_stem(x)
        structure_features, context_features = [], []
        state = None
        for stage_s, stage_c, interaction in zip(
            self.structure_encoder.stages, self.context_encoder.stages, self.interactions
        ):
            fs = stage_s(fs)
            fc = stage_c(fc)
            fs, fc, state = interaction(fs, fc, state)
            structure_features.append(fs)
            context_features.append(fc)
        output_size = x.shape[-2:]
        decoded_s = self.structure_decoder(structure_features, output_size)
        decoded_c = self.context_decoder(context_features, output_size)
        logits_s = self.structure_head(decoded_s)
        logits_c = self.context_head(decoded_c)
        initial_logits = 0.5 * (logits_s + logits_c)
        sdf = self.sdf_head(decoded_s)
        sdf_mu = sdf[:, :1]
        sdf_variance = F.softplus(sdf[:, 1:2]) + 1e-6
        sdf_sigma = sdf_variance.sqrt()
        probabilities_s = F.softmax(logits_s, dim=1)
        probabilities_c = F.softmax(logits_c, dim=1)
        probabilities_0 = F.softmax(initial_logits, dim=1)
        entropy_s = -(probabilities_s * torch.log(probabilities_s.clamp_min(1e-8))).sum(dim=1, keepdim=True)
        entropy_c = -(probabilities_c * torch.log(probabilities_c.clamp_min(1e-8))).sum(dim=1, keepdim=True)
        mixture = 0.5 * (probabilities_s + probabilities_c)
        js = 0.5 * (probabilities_s * (torch.log(probabilities_s.clamp_min(1e-8)) - torch.log(mixture.clamp_min(1e-8)))).sum(1, keepdim=True)
        js = js + 0.5 * (probabilities_c * (torch.log(probabilities_c.clamp_min(1e-8)) - torch.log(mixture.clamp_min(1e-8)))).sum(1, keepdim=True)
        return {
            "structure_features": decoded_s,
            "context_features": decoded_c,
            "structure_logits": logits_s,
            "context_logits": logits_c,
            "initial_logits": initial_logits,
            "probabilities_s": probabilities_s,
            "probabilities_c": probabilities_c,
            "probabilities_0": probabilities_0,
            "sdf_mu": sdf_mu,
            "sdf_sigma": sdf_sigma,
            "sdf_variance": sdf_variance,
            "entropy_s": entropy_s,
            "entropy_c": entropy_c,
            "js_divergence": js,
        }


def otsu_threshold(values: Sequence[float]) -> Optional[float]:
    """Exact Otsu over sorted distinct scores, without histogram quantization.

    Each region has equal weight. Thresholds are midpoints between distinct
    values. First maximizer wins (the smallest threshold).
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Uncertainty scores must be finite")
    values = np.sort(values)
    cuts = np.flatnonzero(values[1:] > values[:-1])
    if cuts.size == 0:
        return None
    cumulative = np.cumsum(values)
    n0 = cuts + 1
    n1 = values.size - n0
    mean0 = cumulative[cuts] / n0
    mean1 = (cumulative[-1] - cumulative[cuts]) / n1
    scores = n0 * n1 / float(values.size ** 2) * (mean0 - mean1) ** 2
    best = cuts[int(np.argmax(scores))]
    return float(values[best] + (values[best + 1] - values[best]) / 2.0)


class ResidualRepairUNet(nn.Module):
    def __init__(self, input_channels: int, feature_channels: int = 64):
        super().__init__()
        self.enc1 = nn.Sequential(ConvNormAct(input_channels, feature_channels), ConvNormAct(feature_channels, feature_channels))
        self.down1 = ConvNormAct(feature_channels, feature_channels, stride=2)
        self.enc2 = nn.Sequential(ConvNormAct(feature_channels, feature_channels), ConvNormAct(feature_channels, feature_channels))
        self.down2 = ConvNormAct(feature_channels, feature_channels, stride=2)
        self.bottleneck = nn.Sequential(ConvNormAct(feature_channels, feature_channels), ConvNormAct(feature_channels, feature_channels))
        self.up1 = nn.ConvTranspose2d(feature_channels, feature_channels, 2, stride=2)
        self.dec1 = ConvNormAct(feature_channels * 2, feature_channels)
        self.up2 = nn.ConvTranspose2d(feature_channels, feature_channels, 2, stride=2)
        self.dec2 = ConvNormAct(feature_channels * 2, feature_channels)
        self.residual = nn.Conv2d(feature_channels, 2, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        u1 = self.up1(b)
        u1 = self.dec1(torch.cat([u1, e2], dim=1))
        u2 = self.up2(u1)
        u2 = self.dec2(torch.cat([u2, e1], dim=1))
        if u2.shape[-2:] != x.shape[-2:]:
            u2 = F.interpolate(u2, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.residual(u2), u2


class OutcomePredictor(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)


class EvidenceGuidedRepairNet(nn.Module):
    """Full paper model with an explicit ``base`` submodule for staged training."""

    def __init__(self, patch_size: int = 16, token_dim: int = 128, attention_dim: int = 64):
        super().__init__()
        self.patch_size = int(patch_size)
        if self.patch_size < 4 or self.patch_size % 4:
            raise ValueError("patch_size must be a positive multiple of 4 (two repair downsampling stages)")
        self.base = DualBranchUncertaintyNet()
        self.token_encoder = nn.Sequential(
            nn.Linear(64 + 6, token_dim), nn.ReLU(inplace=True), nn.Linear(token_dim, token_dim)
        )
        self.query = nn.Linear(token_dim, attention_dim)
        self.key = nn.Linear(token_dim, attention_dim)
        self.value = nn.Linear(token_dim, attention_dim)
        self.repair = ResidualRepairUNet(64 + token_dim + attention_dim, feature_channels=64)
        self.outcome = OutcomePredictor(64 + 64 + token_dim + attention_dim, hidden_channels=64)
        self._stage = None

    def set_stage(self, stage: int):
        """Freeze preceding stages, including their normalization statistics."""
        if stage not in (1, 2, 3):
            raise ValueError("stage must be 1, 2, or 3")
        self._stage = stage
        for module in self.children():
            active = ((module is self.base and stage == 1)
                      or (module is self.outcome and stage == 3)
                      or (module is not self.base and module is not self.outcome and stage == 2))
            for parameter in module.parameters():
                parameter.requires_grad_(active)
        self.train(self.training)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._stage is not None:
            for module in self.children():
                if not any(parameter.requires_grad for parameter in module.parameters()):
                    module.eval()
        return self

    @staticmethod
    def _regions(height: int, width: int, patch_size: int):
        rows = int(math.ceil(height / float(patch_size)))
        cols = int(math.ceil(width / float(patch_size)))
        regions = []
        for row in range(rows):
            for col in range(cols):
                y0, x0 = row * patch_size, col * patch_size
                regions.append((y0, x0, min(patch_size, height - y0), min(patch_size, width - x0)))
        return regions

    def _pool_regions(self, feature, regions):
        height, width = feature.shape[-2:]
        pad_h = (-height) % self.patch_size
        pad_w = (-width) % self.patch_size
        padded = F.pad(feature, (0, pad_w, 0, pad_h))
        valid = F.pad(feature.new_ones((1, 1, height, width)), (0, pad_w, 0, pad_h))
        pooled = F.avg_pool2d(padded, self.patch_size, self.patch_size)
        counts = F.avg_pool2d(valid, self.patch_size, self.patch_size)
        return (pooled / counts).flatten(2).transpose(1, 2)

    def _select_regions(self, base, regions):
        scores = [base["entropy_s"], base["entropy_c"], base["sdf_sigma"]]
        regional_scores = [self._pool_regions(score, regions).squeeze(-1).detach().cpu().numpy() for score in scores]
        selected = []
        for batch_index in range(base["initial_logits"].size(0)):
            chosen = set()
            for regional_batch in regional_scores:
                regional = regional_batch[batch_index]
                threshold = otsu_threshold(regional)
                if threshold is not None:
                    chosen.update(np.flatnonzero(regional > threshold).tolist())
            selected.append(sorted(chosen))
        return selected

    @staticmethod
    def _crop(tensor, y0, x0, valid_h, valid_w, patch_size):
        crop = tensor[..., y0:y0 + valid_h, x0:x0 + valid_w]
        pad_h, pad_w = patch_size - valid_h, patch_size - valid_w
        if pad_h or pad_w:
            crop = F.pad(crop, (0, pad_w, 0, pad_h), mode="replicate")
        return crop

    def forward(self, x, accept: bool = True, return_details: bool = False):
        gradients = torch.is_grad_enabled()
        with torch.set_grad_enabled(gradients and self._stage in (None, 1)):
            base = self.base(x)
        if self._stage == 1:
            return base if return_details else base["probabilities_0"]
        bsz, _, height, width = x.shape
        regions = self._regions(height, width, self.patch_size)
        n_regions = len(regions)
        selected = self._select_regions(base, regions)
        candidate_logits = base["initial_logits"].clone()
        candidate = base["probabilities_0"].clone()
        outcome_values = x.new_zeros((bsz, n_regions, 2))
        selected_mask = torch.zeros((bsz, n_regions), dtype=torch.bool, device=x.device)
        accepted_mask = torch.zeros_like(selected_mask)
        selected_metadata = []
        attention_details = []
        backbone_tokens = state_tokens = tokens = None
        if any(selected):
            # Only stage 2 differentiates candidate generation. Stage 3 uses
            # exactly the same candidates, with fixed parameters and BN stats.
            with torch.set_grad_enabled(gradients and self._stage in (None, 2)):
                fused = torch.cat([base["structure_features"], base["context_features"]], dim=1)
                state = torch.cat([
                    base["sdf_mu"], base["probabilities_0"][:, 1:2], base["entropy_s"], base["entropy_c"],
                    base["sdf_sigma"], base["js_divergence"],
                ], dim=1)
                backbone_tokens = self._pool_regions(fused, regions)
                state_tokens = self._pool_regions(state, regions)
                tokens = self.token_encoder(torch.cat([backbone_tokens, state_tokens], dim=-1))
                keys, values = self.key(tokens), self.value(tokens)
                local_features, local_baselines, local_tokens, local_evidence, local_backbone = [], [], [], [], []
                for batch_index, indices in enumerate(selected):
                    if not indices:
                        continue
                    ids = torch.tensor(indices, dtype=torch.long, device=x.device)
                    queries = self.query(tokens[batch_index, ids])
                    scores = queries.matmul(keys[batch_index].transpose(0, 1)) / math.sqrt(keys.size(-1))
                    scores = scores.scatter(1, ids.unsqueeze(1), float("-inf"))
                    weights = F.softmax(scores, dim=-1)
                    evidence = weights.matmul(values[batch_index])
                    if return_details:
                        attention_details.append((batch_index, ids, weights.detach()))
                    selected_mask[batch_index, ids] = True
                    local_tokens.append(tokens[batch_index, ids])
                    local_evidence.append(evidence)
                    local_backbone.append(backbone_tokens[batch_index, ids])
                    for region_index in indices:
                        y0, x0, valid_h, valid_w = regions[region_index]
                        local_features.append(self._crop(fused[batch_index:batch_index + 1], y0, x0, valid_h, valid_w, self.patch_size))
                        local_baselines.append(self._crop(base["initial_logits"][batch_index:batch_index + 1], y0, x0, valid_h, valid_w, self.patch_size))
                        selected_metadata.append((batch_index, region_index))
                local_tokens = torch.cat(local_tokens)
                local_evidence = torch.cat(local_evidence)
                token_grid = local_tokens[:, :, None, None].expand(-1, -1, self.patch_size, self.patch_size)
                evidence_grid = local_evidence[:, :, None, None].expand(-1, -1, self.patch_size, self.patch_size)
                repair_input = torch.cat([torch.cat(local_features), token_grid, evidence_grid], dim=1)
                residuals, repair_features = self.repair(repair_input)
                local_candidate_logits = torch.cat(local_baselines) + residuals
                local_candidates = F.softmax(local_candidate_logits, dim=1)
                pooled_repair = []
                for local_index, (batch_index, region_index) in enumerate(selected_metadata):
                    y0, x0, valid_h, valid_w = regions[region_index]
                    candidate_logits[batch_index, :, y0:y0 + valid_h, x0:x0 + valid_w] = local_candidate_logits[local_index, :, :valid_h, :valid_w]
                    candidate[batch_index, :, y0:y0 + valid_h, x0:x0 + valid_w] = local_candidates[local_index, :, :valid_h, :valid_w]
                    pooled_repair.append(repair_features[local_index, :, :valid_h, :valid_w].mean(dim=(-2, -1)))
                outcome_input = torch.cat([
                    torch.cat(local_backbone), torch.stack(pooled_repair), local_tokens, local_evidence,
                ], dim=1)
            if self._stage != 2:
                predictions = self.outcome(outcome_input)
                batch_ids, region_ids = zip(*selected_metadata)
                outcome_values[list(batch_ids), list(region_ids)] = predictions
                if accept:
                    accepted_mask = selected_mask & (outcome_values[:, :, 0] > outcome_values[:, :, 1])
        final = base["probabilities_0"].clone()
        if accept and self._stage != 2:
            accepted_cpu = accepted_mask.detach().cpu().numpy()
            for batch_index, region_index in selected_metadata:
                if accepted_cpu[batch_index, region_index]:
                    y0, x0, valid_h, valid_w = regions[region_index]
                    final[batch_index, :, y0:y0 + valid_h, x0:x0 + valid_w] = candidate[batch_index, :, y0:y0 + valid_h, x0:x0 + valid_w]
        result = dict(base)
        result.update({
            "candidate_probabilities": candidate,
            "candidate_logits": candidate_logits,
            "final_probabilities": final,
            "outcome": outcome_values,
            "selected_mask": selected_mask,
            "accepted_mask": accepted_mask,
            "regions": regions,
            "tokens": tokens,
            "backbone_region_features": backbone_tokens,
            "state_region_features": state_tokens,
            "reference_attention": attention_details,
        })
        if return_details:
            return result
        return final


__all__ = [
    "DualBranchUncertaintyNet", "EvidenceGuidedRepairNet", "otsu_threshold",
]
