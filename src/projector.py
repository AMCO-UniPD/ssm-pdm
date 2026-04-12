"""
Python module containing the implementation of the projector
classes
"""

import torch
import torch.nn as nn

class Projector(nn.Module):
    def __init__(self, d_model:int):
        """
        The projector is used to up project the
        input tensor into an higher dimensional space.
        Normally the transformation done at this stage is:
        (B,L,D) → (B,L,H)
        """
        super().__init__()

        self.projector = nn.LazyLinear(d_model)

    def forward(self,x):

        x = self.projector(x) # (B,L,D) → (B,L,H)
        return x
