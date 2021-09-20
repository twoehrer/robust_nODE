#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski (adapted from https://github.com/EmilienDupont/augmented-neural-odes)
"""
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    The dynamics x[k]+f(u[k],x[k]) at given (k, x[k])
    """
    def __init__(self, data_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.mlp(x)
#currently designed for images because there is vector conversion
class ResNet(nn.Module):
    """
    Returns the discrete ResNet semiflow x\mapsto\Phi(x), where
    \Phi might designate the full discrete state of the ResNet, or some projection thereof.
    """
    def __init__(self, data_dim, hidden_dim, num_layers, output_dim=1,
                 is_img=True):
        super(ResNet, self).__init__()
        residual_blocks = \
            [ResidualBlock(data_dim, hidden_dim) for _ in range(num_layers)]
        self.residual_blocks = nn.Sequential(*residual_blocks)
        self.linear_layer = nn.Linear(data_dim, output_dim)
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.is_img = is_img
        self.cross_entropy = is_img

    def forward(self, x, return_features=False):

        traj = []
        traj.append(self.residual_blocks[0](x))      # to store the states/features over layers #changed  x.view(x.size(0),-1) to x
       
        for k in range(1, self.num_layers):
            traj.append(self.residual_blocks[k](traj[k-1]))
        features = self.residual_blocks(x) #changed .view(x.size(0),-1) to x to not have a vector was done for image classification(also above)
        # else:
        #     features = self.residual_blocks(x)
        
        pred = self.linear_layer(features)
        _traj = [self.linear_layer(_) for _ in traj]
                
        if return_features:
            return features, pred
        return pred, _traj, traj  #traj should be the right one   ADD: return of adj_traj

    @property
    def hidden_dim(self):
        return self.residual_blocks.hidden_dim

