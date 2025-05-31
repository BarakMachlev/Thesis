import torch.nn as nn
from .input_embedding import InputEmbedding
from .positional_encoding import PositionalEncoding
from .encoder import Encoder

class EncoderOnlyTransformer(nn.Module):
    def __init__(self, dynamic_input_size=4, metadata_input_size=2, d_model=512, metadata_n_features=32, window_size=32,
                 dropout=0.1, num_encoder_layers=4, h=8):
        super().__init__()

        self.input_embedding = InputEmbedding(
            dynamic_input_size=dynamic_input_size,
            metadata_input_size=metadata_input_size,
            d_model=d_model,
            metadata_n_features=metadata_n_features
        )

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            window_size=window_size
        )

        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            h=h,
            dim_feedforward=2048,
            dropout=dropout
        )

        self.regression_head = nn.Sequential(
            nn.Linear(d_model,1),
            nn.ReLU()  # ensures rain rate ≥ 0
        )

    def forward(self, dynamic_data, metadata):
        """
        dynamic_data: [B, T, 4]
        metadata:     [B, 2]
        Returns:      [B, T, d_model]
        """
        x = self.input_embedding(dynamic_data, metadata)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.regression_head(x)  # [B, T, 1]
        return x