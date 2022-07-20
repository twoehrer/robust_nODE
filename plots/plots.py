#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""
##------------#
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from matplotlib.colors import to_rgb
import imageio

from matplotlib.colors import LinearSegmentedColormap
import os

@torch.no_grad()
def classification_evolution(model, fig_name=None, footnote=None, contour = True, plotlim = [-2, 2]):
    
    
    x1lower, x1upper = plotlim
    x2lower, x2upper = plotlim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fig = plt.figure(figsize=(5, 5), dpi=300)
    
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.figtext(0.5, 0, footnote, ha="center", fontsize=10)

    
   
    model.to(device)

    x1 = torch.arange(x1lower, x1upper, step=0.01, device=device)
    x2 = torch.arange(x2lower, x2upper, step=0.01, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    
    preds, _ = model(model_inputs)
    
    # dim = 2 means that it normalizes along the last dimension, i.e. along the two predictions that are the model output
    m = nn.Softmax(dim=2)
    # softmax normalizes the model predictions to probabilities
    preds = m(preds)

    #we only need the probability for being in class1 (as prob for class2 is then 1- class1)
    preds = preds[:, :, 0]
    preds = preds.unsqueeze(2)  # adds a tensor dimension at position 2
    
    plt.grid(False)
    plt.xlim([x1lower, x1upper])
    plt.ylim([x2lower, x2upper])

    ax = plt.gca()
    ax.set_aspect('equal') 
    
    if contour:
        colors = [to_rgb("C1"), [1, 1, 1], to_rgb("C0")] # first color is orange, last is blue
        cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=40)
        z = np.array(preds).reshape(xx1.shape)
        
        levels = np.linspace(0.,1.,8).tolist()
        
        cont = plt.contourf(xx1, xx2, z, levels, alpha=1, cmap=cm, zorder = 0, extent=(x1lower, x1upper, x2lower, x2upper)) #plt.get_cmap('coolwarm')
        cbar = fig.colorbar(cont, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('prediction prob.')
    

    if fig_name:
        plt.savefig(fig_name + '.png', bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
        plt.clf()
        plt.close()
        
def loss_evolution(trainer, epoch, filename = '', figsize = None):

    fig = plt.figure(dpi = 300, figsize=(figsize))
    labelsize = 10

    #plot whole loss history in semi-transparent
    plt.plot(trainer.histories['epoch_loss_history'], alpha = 0.5)
    plt.plot(trainer.histories['epoch_loss_rob_history'], '--', zorder = -1, alpha = 0.5)
    
    if trainer.eps > 0: #if the trainer has a robustness term
        standard_loss_term = [loss - rob for loss, rob in zip(trainer.histories['epoch_loss_history'],trainer.histories['epoch_loss_rob_history'])]
        plt.plot(standard_loss_term,'--', alpha = 0.5)
        leg = plt.legend(['total loss', 'gradient term', 'standard term'], prop= {'size': labelsize})
    else: leg = plt.legend(['standard loss', '(inaktive) gradient term'], prop= {'size': labelsize})
        
    #set alpha to 1
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    plt.plot(trainer.histories['epoch_loss_history'][0:epoch+1], color = 'C0')
    plt.scatter(epoch, trainer.histories['epoch_loss_history'][epoch])
    
    plt.plot(trainer.histories['epoch_loss_rob_history'][0:epoch +1], '--', color = 'C1')
    plt.scatter(epoch, trainer.histories['epoch_loss_rob_history'][epoch], color = 'C1')
    
    if trainer.eps > 0: #if the trainer has a robustness term
        plt.plot(standard_loss_term[0:epoch+1],'--', color = 'C2')
        plt.scatter(epoch, standard_loss_term[epoch], color = 'C2')
        
    plt.xlim(0, len(trainer.histories['epoch_loss_history']) - 1)
    # plt.ylim([0,0.75])
    plt.yticks(np.arange(0,1,0.25))
    plt.grid(zorder = -2)
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.set_aspect('auto')
    if trainer.eps > 0:
        plt.ylabel('Loss Robust', size = labelsize)
        
    else:
        plt.ylabel('Loss Standard', size = labelsize)


    if not filename == '':
        plt.savefig(filename + '.png', bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
        plt.clf()
        plt.close()
        
    else:
        plt.show()
        print('no filename given')
        

def comparison_plot(fig1, title1, fig2, title2, output_file, figsize = (10,10), show = False):
    plt.figure(dpi = 300, figsize=figsize)
    plt.subplot(121)
    sub1 = imageio.imread(fig1)
    plt.imshow(sub1)
    plt.title(title1)
    plt.axis('off')

    plt.subplot(122)
    sub2 = imageio.imread(fig2)
    plt.imshow(sub2)
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=600, format='png', facecolor = 'white')
    if show: plt.show()
    else:
        plt.gca()
        plt.close()
        
        
def train_to_classifier_imgs(model, trainer, dataloader, subfolder, num_epochs, plotfreq, filename = '', plotlim = [-2, 2]):
    
    if not os.path.exists(subfolder):
            os.makedirs(subfolder)

    fig_name_base = os.path.join(subfolder,'') #os independent file path

    for epoch in range(0,num_epochs,plotfreq):
        print(f'\n{epoch =}')
        trainer.train(dataloader, plotfreq)
        
        classification_evolution(model, fig_name = fig_name_base + filename + str(epoch), footnote = f'{epoch = }', plotlim = plotlim)