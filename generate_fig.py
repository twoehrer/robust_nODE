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
from models.training import Trainer
from models.neural_odes import NeuralODE
from models.resnets import ResNet
import pickle


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
T, num_steps = 5.0, 10
dt = T/num_steps
turnpike = False
bound = 0.
fp = False
cross_entropy = True

if turnpike:
    weight_decay = 0 if bound>0. else dt*0.01
else: 
    weight_decay = dt*0.01          #0.01 for fp, 0.1 else

anode = NeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='tanh', 
                    architecture='inside', T=T, time_steps=num_steps, fixed_projector=fp, cross_entropy=cross_entropy)


# anode = ResNet(data_dim, hidden_dim, num_steps) #this is a try

optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3, weight_decay=weight_decay) #weight decay parameter modifies norm
trainer_anode = Trainer(anode, optimizer_anode, device, cross_entropy=cross_entropy, 
                        turnpike=turnpike, bound=bound, fixed_projector=fp)
num_epochs = 50 #number of optimization epochs for gradient decent (?) lower number means worse optimization
visualize_features = True #changed

import time
start_time = time.time()
if visualize_features:
    feature_history = get_feature_history(trainer_anode, dataloader, 
                                          inputs, targets, num_epochs)
else:
    trainer_anode.train(dataloader, num_epochs)
print("--- %s seconds ---" % (time.time() - start_time))

##--------------#
## Plots:
plt_norm_state(anode, inputs, timesteps=num_steps)
plt_train_error(anode, inputs, targets, timesteps=num_steps)
feature_plot(feature_history, targets)
plt_classifier(anode, num_steps=10)  #numsteps defines the amount of pictures created for the gif
trajectory_gif(anode, inputs, targets, timesteps=num_steps)
# ##--------------#
## Saving the weights:
# pars = []
# for param_tensor in anode.state_dict():
#    pars.append(anode.state_dict()[param_tensor])
#    #print(param_tensor, "\t", anode.state_dict()[param_tensor])

# with open("plots/controls.txt", "wb") as fp:
#    pickle.dump(pars, fp)
# plt_norm_control(anode)
# ##--------------#
