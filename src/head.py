"""
Python module containing the implementation of the model head
blocks for the RUL models
"""

import ipdb
import torch.nn as nn
import torch.nn.functional as F

from exp_config import ModelConfig


class MonotonicLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        pre_activation=nn.Identity(),
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.act = pre_activation

    def forward(self, x):
        w_pos = self.weight.clamp(min=0.0)
        w_neg = self.weight.clamp(max=0.0)
        x_pos = F.linear(self.act(x), w_pos, self.bias)
        x_neg = F.linear(self.act(-x), w_neg, self.bias)
        return x_pos + x_neg


class Head(nn.Module):
    def __init__(self, config: ModelConfig, d_output: int):
        """
        The head is responsible for combining the features
        extracted throught the extractor and perform the downstream
        task. In this case predicting the RUL signal.
        Normally the transformation done at this stage is:
        (B,L,H) → (B,O)
        """
        super().__init__()
        self.config = config
        self.d_output = d_output

        self.decoder = nn.LazyLinear(self.d_output)

    def forward(self, x):

        x = x[:, -1, :].squeeze(1)  # (B, L, H) -> (B,1,H) → (B, H)
        x = self.decoder(x)  # (B,H) → (B,O)

        return x


class QuantileHead(Head):
    def __init__(self, tau: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau

    def forward(self, x):
        if getattr(self, "debug_tau", False):
            print(f"[{type(self).__name__}] tau={self.tau}")
        x = x[:, -1, :].squeeze(1)  # (B, L, H) -> (B,1,H) → (B, H)
        # (B,H) → (B,O)
        x = self.decoder(
            x) * self.tau if self.config.tau_mult else self.decoder(x)

        return x


class QuantileScaleHead(Head):
    """
    Quantile head that predicts a median and separate positive scales for the
    lower and upper halves of the conditional distribution.

    For a fixed representation, the output is monotonic in ``tau``:

        Q(tau) = median + (2 * tau - 1) * scale

    where the lower scale is used below the median and the upper scale above
    it. Unlike :class:`QuantileHead`, this head does not rely on ``tau_feat`` or
    ``tau_mult``.
    """

    def __init__(self, tau: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be between 0 and 1, got {tau}")

        self.tau = tau
        self.lower_scale_decoder = nn.LazyLinear(self.d_output)
        self.upper_scale_decoder = nn.LazyLinear(self.d_output)

    def forward(self, x):
        if getattr(self, "debug_tau", False):
            print(f"[{type(self).__name__}] tau={self.tau}")
        x = x[:, -1, :].squeeze(1)  # (B, L, H) -> (B, H)

        median = self.decoder(x)
        lower_scale = F.softplus(self.lower_scale_decoder(x))
        upper_scale = F.softplus(self.upper_scale_decoder(x))

        centered_tau = 2.0 * self.tau - 1.0
        scale = lower_scale if self.tau < 0.5 else upper_scale
        return median + centered_tau * scale


class MonotonicHead(Head):
    def __init__(self, n_mono_neurons: int = 3,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.act == "gelu":
            self.activation = nn.GELU()
        elif self.config.act == "relu":
            self.activation = nn.ReLU()
        elif self.config.act == "silu":
            self.activation = nn.SiLU()
        elif self.config.act == "selu":
            self.activation = nn.SELU()
        elif self.config.act == "celu":
            self.activation = nn.CELU()
        else:
            self.activation = nn.Identity()

        n_neurons = self.config.d_model + n_mono_neurons

        self.mono = nn.ModuleList([
            MonotonicLinear(
                n_neurons, n_neurons, pre_activation=nn.Identity()),
            *[MonotonicLinear(n_neurons, n_neurons,
                              pre_activation=self.activation) for _ in range(self.config.n_mono_layers)],
            MonotonicLinear(n_neurons,
                            self.d_output, pre_activation=self.activation)
        ])

        def forward(self, x):

            x = x[:, -1, :].squeeze(1)  # (B, L, H) -> (B,1,H) → (B, H)

            for layer in self.mono:
                x = layer(x)

            return x


class MonoQuantileHead(MonotonicHead):

    def __init__(self, tau: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau

    def forward(self, x):

        x = x[:, -1, :].squeeze(1)  # (B, L, H) -> (B,1,H) → (B, H)

        for layer in self.mono:
            x = layer(x)

        x = x * self.tau if self.config.tau_mult else x
        return x
