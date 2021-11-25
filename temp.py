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

print('requires grad', rob_node.adj_traj_p1.requires_grad)

rob_node.adj_traj_p1 = False

print('requires grad', rob_node.adj_traj_p1.requires_grad)


print('matmul', rob_node.adj_traj_p1[-1].matmul(rob_node.adj_traj_p1[-1]))

print('parameters',rob_node.parameters())

# optimizer_node = torch.optim.Adam(rob_node.parameters(), lr=1e-3, weight_decay = 0) #weight decay parameter modifies norm

# trainer_rob_node = Trainer(rob_node, optimizer_node, device, cross_entropy=cross_entropy, 
#                         turnpike=turnpike, bound=bound, fixed_projector=fp)

if turnpike:
    weight_decay = 0 if bound>0. else dt*0.01
else: 
    weight_decay = dt*0.01          #0.01 for fp, 0.1 else

optimizer_rob_node = torch.optim.Adam(rob_node.parameters(), lr=1e-3, weight_decay=weight_decay) #weight decay parameter modifies norm
trainer_rob_node = robTrainer(rob_node, optimizer_rob_node, device, cross_entropy=cross_entropy, 
                        turnpike=turnpike, bound=bound, fixed_projector=fp)

num_epochs = 70

trainer_rob_node.train(dataloader, num_epochs)

trajectory_gif(rob_node, inputs, targets, timesteps=num_steps, filename='trajectory_rob.gif')


