#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2019-2020 Apple Inc. All Rights Reserved.
#
import torch.nn as nn


class IndependentVectorizedApproximator(nn.Module):
    def __init__(self, latent_size, activation=nn.Tanh):
        """ A vectorized approximator that takes each input (feature-elem by feature-elem) and produces an approximation.
            This allows for a network that shares parameters and enables an easier approximation.

        :param latent_size: latent size for model
        :returns: IndependentVectorizedApproximator
        :rtype: nn.Module

        """
        super(IndependentVectorizedApproximator, self).__init__()

        # the actual model
        self.approximator = nn.Sequential(
                nn.Linear(1, latent_size),
                activation(),
                nn.Linear(latent_size, latent_size),
                activation(),
                nn.Linear(latent_size, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.approximator(x).squeeze(-1)
