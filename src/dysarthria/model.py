"""WavLM encoder with frame-level Mixture-of-Experts routing."""

from __future__ import annotations

try:
    import torch
    from torch import nn
    from transformers import WavLMModel
except ImportError as exc:  # pragma: no cover - exercised only without model extras
    raise ImportError('Install model dependencies with: pip install -e ".[model]"') from exc


class WavLMMixtureOfExperts(nn.Module):
    """Binary utterance classifier with transparent expert routing weights."""

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        *,
        num_experts: int = 4,
        expert_dim: int = 256,
        dropout: float = 0.1,
        freeze_feature_encoder: bool = True,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be at least 2")
        self.encoder = WavLMModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        if freeze_feature_encoder:
            self.encoder.freeze_feature_encoder()

        self.gate = nn.Sequential(
            nn.Linear(hidden, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim, num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, expert_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_dim, expert_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.classifier = nn.Linear(expert_dim, 2)

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None):
        encoded = self.encoder(input_values=input_values, attention_mask=attention_mask)
        frames = encoded.last_hidden_state
        routing = torch.softmax(self.gate(frames), dim=-1)
        expert_outputs = torch.stack([expert(frames) for expert in self.experts], dim=-2)
        mixed = torch.sum(routing.unsqueeze(-1) * expert_outputs, dim=-2)

        if attention_mask is None:
            pooled = mixed.mean(dim=1)
        else:
            feature_mask = self.encoder._get_feature_vector_attention_mask(
                mixed.shape[1], attention_mask
            ).unsqueeze(-1)
            feature_mask = feature_mask.to(mixed.dtype)
            pooled = (mixed * feature_mask).sum(dim=1) / feature_mask.sum(dim=1).clamp_min(1)
        return {"logits": self.classifier(pooled), "routing_weights": routing}
