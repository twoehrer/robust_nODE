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
import sys
import matplotlib.pyplot as plt
import imageio



##--------------#
## Data: 
with open('data.txt', 'rb') as fp:
    data_line, test = pickle.load(fp)
dataloader = DataLoader(data_line, batch_size=64, shuffle=False)
dataloader_viz = DataLoader(data_line, batch_size=128, shuffle=False)
for inputs, targets in dataloader_viz:
    break


##--------------#
## Setup:
hidden_dim, data_dim = 2, 2
T, num_steps = 5.0, 10  #T is the end time, num_steps are the amount of discretization steps for the ODE solver
dt = T/num_steps
turnpike = True
bound = 0.
fp = False
cross_entropy = True



num_epochs = 140 #number of optimization epochs for gradient decent

if turnpike:
    weight_decay = 0 if bound>0. else dt*0.01
else: 
    weight_decay = dt*0.01          #0.01 for fp, 0.1 else

anode = NeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='tanh', 
                    architecture='outside', T=T, time_steps=num_steps, fixed_projector=fp, cross_entropy=cross_entropy)



rob_node = robNeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='tanh', 
                            architecture='outside', T=T, time_steps=num_steps, fixed_projector=fp, cross_entropy=cross_entropy)

optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3, weight_decay=weight_decay) #weight decay parameter modifies norm
trainer_anode = Trainer(anode, optimizer_anode, device, cross_entropy=cross_entropy, 
                        turnpike=turnpike, bound=bound, fixed_projector=fp)


optimizer_rob_node = torch.optim.Adam(rob_node.parameters(), lr=1e-3, weight_decay=weight_decay) #weight decay parameter modifies norm
trainer_rob_node = robTrainer(rob_node, optimizer_rob_node, device, cross_entropy=cross_entropy, 
                        turnpike=turnpike, bound=bound, fixed_projector=fp)                        


print("Model's state_dict:")
for param_tensor in rob_node.state_dict():
    print(param_tensor, "\t", rob_node.state_dict()[param_tensor].size())


# sys.exit()

visualize_features = False #changed

import time
start_time = time.time()
# if visualize_features:
#     feature_history = get_feature_history(trainer_anode, dataloader, 
#                                           inputs, targets, num_epochs)
# else:

trainer_anode.train(dataloader, num_epochs)
trainer_rob_node.train(dataloader, num_epochs)
print("--- %s seconds ---" % (time.time() - start_time))

##--------------#
# ## Plots:
# plt_norm_state(anode, inputs, timesteps=num_steps)
# plt_train_error(anode, inputs, targets, timesteps=num_steps)

# plt_norm_state(rob_node, inputs, timesteps=num_steps)
# plt_train_error(rob_node, inputs, targets, timesteps=num_steps, filename = 'rob_train_error.pdf')
# feature_plot(feature_history, targets)



filename_base = '1traj'
filename_s = filename_base + '_s'
filename_r = filename_base + '_r'

plt_classifier(anode, num_steps=10, save_fig = '1generalization.pdf') 
plt_classifier(rob_node, num_steps=10, save_fig = '1rob_generalization.pdf')
trajectory_gif(anode, inputs, targets, timesteps=num_steps, filename = filename_s +'.gif')
trajectory_gif(rob_node, inputs, targets, timesteps=num_steps, filename = filename_r + '.gif')

plt.figure(1)
plt.subplot(121)
plt.imshow(imageio.imread(filename_s + '29.png'))
plt.axis('off')
plt.subplot(122)
plt.imshow(imageio.imread(filename_r + '29.png'))
plt.axis('off')

plt.savefig('1comparison_' + filename_base + '.png',
                    format='png', dpi=300, bbox_inches='tight')

# inputs[0] = inputs[0] + 2



# trajectory_gif(anode, inputs, targets, timesteps=num_steps, filename = '1standard_pert.gif')
# trajectory_gif(rob_node, inputs, targets, timesteps=num_steps, filename = '1rob_pert.gif')

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



