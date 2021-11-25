#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""
##--------------#
import torch
device = torch.device('cpu')
from torch.utils.data import DataLoader
from plots.gifs import trajectory_gif
from plots.plots import get_feature_history, plt_train_error, plt_norm_state, plt_norm_control, plt_classifier, feature_plot, plt_dataset
from models.training import Trainer, robTrainer
from models.neural_odes import NeuralODE, robNeuralODE
from models.resnets import ResNet
import pickle

from models.neural_odes import Dynamics, adj_Dynamics, Semiflow

##--------------#
## Data: 
with open('data.txt', 'rb') as fp:
    data_line, test = pickle.load(fp)
dataloader = DataLoader(data_line, batch_size=64, shuffle=True)
dataloader_viz = DataLoader(data_line, batch_size=128, shuffle=True)
for inputs, targets in dataloader_viz:
    break

##--------------#
## Setup:
hidden_dim, data_dim = 2, 2
T, num_steps = 5.0, 15
dt = T/num_steps
turnpike = False
bound = 0.
fp = False
cross_entropy = True

time_steps = num_steps


anode = NeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='tanh', 
                    architecture='inside', T=T, time_steps=num_steps, fixed_projector=fp, cross_entropy=cross_entropy)

dynamics = Dynamics(device, data_dim, hidden_dim, augment_dim=0, non_linearity='tanh', 
                    architecture='inside')
x = torch.tensor([1.,1.])
x_traj = anode.flow.trajectory(x, num_steps)                    
adj_dynamics = adj_Dynamics(dynamics, x_traj, device, data_dim, hidden_dim, augment_dim=0, non_linearity='tanh')

flow = Semiflow(device, dynamics)
adj_flow = Semiflow(device, adj_dynamics)


print('dynamics: ',dynamics(1.,x))

print('adj_dynamics:', adj_dynamics(1., x + 4))

print('adj_flow:', adj_flow.trajectory(x,num_steps))

rob_node = robNeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='tanh', 
                    architecture='outside', T=T, time_steps=num_steps, fixed_projector=fp, cross_entropy=cross_entropy)
pred, traj = rob_node(x)

p1 = torch.tensor([1.,0.])
p2 = torch.tensor([0.,1.])
proj_adj_traj_p1 = adj_flow.trajectory(p1, time_steps)
proj_adj_traj_p2 = adj_flow.trajectory(p2, time_steps)

print('proj_adj_traj 1',rob_node.adj_traj_p1)
rob_node(x+1)

print('proj_adj_traj 2',rob_node.adj_traj_p1)

with open('data.txt', 'rb') as fp:
    data_line, test = pickle.load(fp)

    print(data_line)



trajectory_gif(anode, inputs, targets, timesteps=num_steps, filename = '1standard.gif')
trajectory_gif(rob_node, inputs, targets, timesteps=num_steps, filename = '1rob.gif')

inputs[0] = inputs[0] + 2

trajectory_gif(anode, inputs, targets, timesteps=num_steps, filename = '1standard_pert.gif')
trajectory_gif(rob_node, inputs, targets, timesteps=num_steps, filename = '1rob_pert.gif')