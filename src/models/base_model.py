import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TimesformerConfig, TimesformerModel


class AnomalyModel(nn.Module):
    def __init__(self, num_classes: int = 14, seq_len: int = 16, cache_dir: str = "./pretrained_cache"):
        super().__init__()

        os.makedirs(cache_dir, exist_ok=True)

        # --- Step 1: Load config with desired frames ---
        self.config = TimesformerConfig.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_frames=seq_len,
            cache_dir=cache_dir
        )

        # --- Step 2: Create empty Timesformer backbone ---
        backbone = TimesformerModel(self.config)

        # --- Step 3: Cache pretrained weights ---
        weights_path = os.path.join(cache_dir, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            print("Downloading Timesformer pretrained weights...")
            torch.hub.download_url_to_file(
                "https://huggingface.co/facebook/timesformer-base-finetuned-k400/resolve/main/pytorch_model.bin",
                weights_path
            )

        # --- Step 4: Load state dict ---
        state_dict = torch.load(weights_path, map_location="cpu")

        # --- Step 5: Remove classifier weights (we add our own head) ---
        state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}

        # --- Step 6: Auto-detect and interpolate time embeddings if needed ---
        te_key = "embeddings.time_embeddings"
        if te_key in state_dict:
            pretrained_emb = state_dict[te_key]  # shape: [1, old_len, hidden]
            old_len = pretrained_emb.shape[1]

            if old_len != seq_len:
                new_emb = F.interpolate(
                    pretrained_emb.permute(0, 2, 1),  # to [1, hidden, old_len]
                    size=seq_len,
                    mode="linear",
                    align_corners=False
                ).permute(0, 2, 1)  # back to [1, seq_len, hidden]
                state_dict[te_key] = new_emb
                print(f"[Embedding Fix] Interpolated time embeddings {old_len} → {seq_len}")

        # --- Step 7: Load weights cleanly ---
        missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"[Warning] Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"[Warning] Unexpected keys: {unexpected_keys}")

        self.backbone = backbone
        self.backbone.train()
        self.backbone.gradient_checkpointing_enable()

        # --- Step 8: Heads ---
        self.ts_fc = nn.Sequential(
            nn.Linear(self.config.hidden_size, num_classes),
            nn.Dropout(0.3)
        )

        self.yolo_fc = nn.Sequential(
            nn.Linear(80, num_classes),
            nn.Dropout(0.3)
        )

        self.gate = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.Dropout(0.3)
        )

    def forward(self, x: torch.Tensor, yolo_features: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5 or x.size(2) != 3:
            raise ValueError(f"Expected input shape [batch, seq_len, 3, height, width], got {x.shape}")
        if yolo_features.dim() != 3 or yolo_features.size(2) != 80:
            raise ValueError(f"Expected yolo_features shape [batch, seq_len, 80], got {yolo_features.shape}")

        ts_outputs = self.backbone(pixel_values=x).last_hidden_state
        ts_logits = self.ts_fc(ts_outputs[:, 0, :])

        yolo_logits = self.yolo_fc(yolo_features.mean(dim=1))

        gate = self.gate(torch.cat([ts_logits, yolo_logits], dim=1))
        fused_logits = self.fusion(gate * ts_logits + (1 - gate) * yolo_logits)

        return fused_logits

    def detect_anomaly(self, scores: torch.Tensor, thresh: float = 0.5) -> str:
        scores = torch.softmax(scores, dim=1)
        pred = torch.argmax(scores, dim=1)
        conf = torch.max(scores, dim=1).values

        if pred.item() == 7:
            return "No Anomaly"
        elif conf.item() < thresh:
            return "Unknown Anomaly"
        else:
            classes = [
                'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
                'Fighting', 'Normal Videos', 'RoadAccidents', 'Robbery',
                'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
            ]
            return classes[pred.item()]
