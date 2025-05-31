import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, dynamic_input_size, metadata_input_size, d_model, metadata_n_features):
        super().__init__()

        self.dynamic_n_features = d_model - metadata_n_features
        self.metadata_n_features = metadata_n_features

        self.dynamic_linear = nn.Linear(dynamic_input_size, self.dynamic_n_features)
        self.metadata_linear = nn.Linear(metadata_input_size, self.metadata_n_features)

    def forward(self, dynamic_data, metadata):
        """
        dynamic_data: [B, T, 4]
        metadata:     [B, 2]
        Returns:      [B, T, d_model]
        """
        B, T, _ = dynamic_data.shape

        # Project dynamic data: [B, T, 4] → [B, T, d_model - metadata_n_features]
        dyn_emb = self.dynamic_linear(dynamic_data)

        # Project metadata: [B, 2] → [B, metadata_n_features] → [B, T, metadata_n_features]
        meta_emb = self.metadata_linear(metadata)           # [B, metadata_n_features]
        meta_emb = meta_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, metadata_n_features]

        # Concatenate along feature dimension
        x = torch.cat([dyn_emb, meta_emb], dim=-1)          # [B, T, d_model]
        return x
