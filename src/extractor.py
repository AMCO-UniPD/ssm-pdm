"""
Python module containing the implementation of the feature extractor
blocks for the RUL models
"""

import ipdb
import torch
import torch.nn as nn

# s5 imports
from s5 import S5Block

# transformer imports
from transformer_encoder import TransformerEncoder
from transformer_encoder.utils import PositionalEncoding

# informer imports
from informer import *
from exp_config import ModelConfig

# s4 imports
from s4 import DropoutNd
from s4 import S4Block as S4

# s4d imports
from s4d import S4D


class Extractor(nn.Module):
    def __init__(self, config: ModelConfig):
        """
        The extractor is responsible for extracting the
        temporal features from the high dimensional space.
        Normally the transformation done at this stage is:
        (B,L,H) → (B,L,H)
        """
        super().__init__()

        self.config = config
        self.hidden_layers = nn.ModuleList()
        self.acts = nn.ModuleList()

        if config.act == "gelu":
            self.activation = nn.GELU()
        elif config.act == "relu":
            self.activation = nn.ReLU()
        elif config.act == "silu":
            self.activation = nn.SiLU()
        elif config.act == "selu":
            self.activation = nn.SELU()
        else:
            self.activation = nn.Identity()

        # Most specialized extractors override ``forward`` and build their own
        # layers. Registering the generic LazyLinear stack for those classes
        # leaves parameters that can never be initialized by a model forward
        # pass, which in turn breaks parameter counters and profilers.
        if type(self).forward is Extractor.forward:
            for _ in range(self.config.n_layers):
                self.hidden_layers.append(nn.LazyLinear(self.config.d_model))
                self.acts.append(self.activation)

    def forward(self, x):

        # In this loop it's all (B,L,H) → (B,L,H)
        for layer, act in zip(self.hidden_layers, self.acts):

            x = layer(x)
            x = act(x)

        return x


class S4Extractor(Extractor):
    """
    S4 model feature extractor
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prenorm = self.config.prenorm
        self.device = self.config.device
        self.d_model = self.config.d_model
        self.n_layers = self.config.n_layers
        self.dropout = self.config.dropout
        self.act = self.config.activation
        self.gate_act = self.config.gate_act
        self.mult_act = self.config.mult_act
        self.final_act = self.config.final_act

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(self.n_layers):
            self.s4_layers.append(
                S4(
                    self.d_model,
                    dropout=self.dropout,
                    activation=self.act,
                    gate_act=self.gate_act,
                    mult_act=self.mult_act,
                    final_act=self.final_act,
                    transposed=True,
                    lr=min(0.001, self.config.lr),
                )
            )
            self.norms.append(nn.LayerNorm(self.d_model))
            self.dropouts.append(nn.Dropout(self.dropout))

    def forward(self, x):

        x = x.transpose(-1, -2)  # (B, L, H) -> (B, H, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, H, L) -> (B, H, L)

            z = x

            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)  # (B, H, L) -> (B, L, H)

        return x


class S4DExtractor(Extractor):
    """
    S4D model feature extractor
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = self.config.device
        self.d_state = self.config.d_state
        self.act = self.config.act
        self.d_model = self.config.d_model
        self.n_layers = self.config.n_layers
        self.dropout = self.config.dropout

        # Stack S4D layers as residual blocks
        self.s4d_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(self.n_layers):
            self.s4d_layers.append(
                S4D(
                    d_model=self.d_model,
                    d_output=self.d_model,
                    d_state=self.d_state,
                    dropout=self.dropout,
                    act=self.act,
                    transposed=True,
                )
            )
            self.norms.append(nn.LayerNorm(self.d_model))
            self.dropouts.append(nn.Dropout(self.dropout))

    def forward(self, x):

        x = x.transpose(-1, -2)  # (B, L, H) -> (B, H, L)
        for layer, norm, dropout in zip(self.s4d_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, H, L) -> (B, H, L)

            z = x

            # Apply S4D block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4D block
            z = dropout(z)

            # Residual connection
            x = z + x

            # layer norm
            x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        return x


class S5Extractor(Extractor):
    """
    S5 model feature extractor
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = self.config.device

        # Stack S5 layers as residual blocks
        self.s5_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(self.config.n_layers):
            self.s5_layers.append(
                S5Block(
                    dim=self.config.d_model,
                    state_dim=self.config.d_state,
                    bidir=self.config.bidir,
                    ff_dropout=self.config.ff_dropout,
                    attn_dropout=self.config.attn_dropout,
                )
            )
            self.norms.append(nn.LayerNorm(self.config.d_model))
            self.dropouts.append(nn.Dropout(self.config.dropout))

    def forward(self, x):

        for layer, norm, dropout in zip(
            self.s5_layers, self.norms, self.dropouts
        ):  # (B, L, H) -> (B, L, H). The P is used inside here (black box we do not care)

            # 1. layer
            x = layer(x)

            if torch.isnan(x).any():
                print("-" * 50)
                print(
                    "Obtained NaN after x=layer(x) operation, check better inside the S5Block module"
                )
                print("-" * 50)
                ipdb.set_trace()

            # 2. dropout
            x = dropout(x)
            # 3. norm
            x = norm(x)

        return x


class LinearExtractor(Extractor):
    """
    Linear model feature extractor
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # NOTE: In the Linear extractor we do not need
        # to extract the features so the extractor block
        # is the identity block
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.identity(x)
        return x

class MLPExtractor(Extractor):
    """
    MLP model feature extractor
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.acts = nn.ModuleList()

        if self.config.act == "gelu":
            self.activation = nn.GELU()
        elif self.config.act == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_model),
                nn.LayerNorm(self.config.d_model),
                self.activation,
                nn.Dropout(self.config.dropout)
            )
            for _ in range(self.config.n_layers)
        ])

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x


class RecurrentExtractor(Extractor):
    """
    Recurrent model (i.e. LSTM, RNN, GRU) feature extractor
    """

    def __init__(self, model_name: str = "LSTM", *args, **kwargs):

        super().__init__(*args, **kwargs)

        if model_name == "LSTM":
            self.recurrent = nn.LSTM(
                input_size=self.config.d_model,
                hidden_size=self.config.d_model,
                num_layers=self.config.n_layers,
                batch_first=True,
                dropout=self.config.dropout,
            )
        elif model_name == "GRU":
            self.recurrent = nn.GRU(
                input_size=self.config.d_model,
                hidden_size=self.config.d_model,
                num_layers=self.config.n_layers,
                batch_first=True,
                dropout=self.config.dropout,
            )
        elif model_name == "RNN":
            self.recurrent = nn.RNN(
                input_size=self.config.d_model,
                hidden_size=self.config.d_model,
                num_layers=self.config.n_layers,
                batch_first=True,
                dropout=self.config.dropout,
            )

        self.norm = nn.LayerNorm(self.config.d_model)


class TransformerExtractor(Extractor):
    """
    Transformer model feature extractor
    """

    def __init__(self, output_size:int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_size = output_size

        self.embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings=self.config.d_model, embedding_dim=self.config.d_model
            ),
            PositionalEncoding(
                d_model=self.config.d_model,
                dropout=self.config.dropout,
                max_len=self.output_size,
            ),
        )

        self.encoder = TransformerEncoder(
            d_model=self.config.d_model,
            d_ff=self.config.d_ff,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
        )

    def forward(self, x):

        mask = torch.zeros(x.size(0), x.size(1)).to(x.device)
        x = x.argmax(dim=-1)  # (B, L, D) -> (B, L)
        x = self.embedding(x)  # (B, L) -> (B, L, H)
        x = self.encoder(x, mask)  # (B, L, H) -> (B, L, H)

        return x


class InformerExtractor(Extractor):
    """
    Informer model feature extractor
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Encoding
        self.enc_embedding = DataEmbedding(
            c_in=self.config.d_model, d_model=self.config.d_model, dropout=self.config.dropout
        )
        # Attention
        Attn = ProbAttention if self.config.attn == "prob" else FullAttention
        # Encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    attention=AttentionLayer(
                        attention=Attn(
                            mask_flag=False,
                            factor=self.config.factor,
                            attention_dropout=self.config.dropout,
                            output_attention=True,
                        ),
                        d_model=self.config.d_model,
                        n_heads=self.config.n_heads,
                        mix=False,
                    ),
                    d_model=self.config.d_model,
                    d_ff=self.config.d_ff,
                    dropout=self.config.dropout,
                    activation=self.config.inf_activation,
                )
                for _ in range(self.config.n_layers)
            ],
            conv_layers=(
                [
                    ConvLayer(self.config.d_model)
                    for _ in range(self.config.n_layers - 1)
                ]
                if self.config.distil
                else None
            ),
            norm_layer=torch.nn.LayerNorm(self.config.d_model),
        )

    def forward(self, x):

        enc_out = self.enc_embedding(x)  # [B,L,D] -> [B,L,H]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # [B,L,H] -> [B,L,H]

        return enc_out
