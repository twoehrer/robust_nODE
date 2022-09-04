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
def visualize_classification(model, data, label, grad = None, fig_name=None, footnote=None, contour = True, x1lims = [-2, 2], x2lims = [-2, 2]):
    
    
    x1lower, x1upper = x1lims
    x2lower, x2upper = x2lims

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    fig = plt.figure(figsize=(5, 5), dpi=100)
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0", zorder = 1)
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1", zorder = 1)

    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.figtext(0.5, 0, footnote, ha="center", fontsize=10)
    # plt.legend()
    if not grad == None:
        for i in range(len(data[:, 0])):
            plt.arrow(data[i, 0], data[i, 1], grad[i, 0], grad[i, 1],
                    head_width=0.05, head_length=0.1, fc='k', ec='k', alpha=0.5, length_includes_head = True)

   
    model.to(device)
    # creates the RGB values of the two scatter plot colors.
    # c0 = torch.Tensor(to_rgba("C0")).to(device)
    # c1 = torch.Tensor(to_rgba("C1")).to(device)

    

    x1 = torch.arange(x1lower, x1upper, step=0.01, device=device)
    x2 = torch.arange(x2lower, x2upper, step=0.01, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    preds, _ = model(model_inputs)
    # dim = 2 means that it normalizes along the last dimension, i.e. along the two predictions that are the model output
    m = nn.Softmax(dim=2)
    # softmax normalizes the model predictions to probabilities
    preds = m(preds)

    # now we only want to have the probability for being in class1 (as prob for class2 is then 1- class1)
    preds = preds[:, :, 0]
    preds = preds.unsqueeze(2)  # adds a tensor dimension at position 2
    # Specifying "None" in a dimension creates a new one. The rgb values hence get rescaled according to the prediction
    # output_image = (1 - preds) * c1[None, None] + preds * c0[None, None]
    # # Convert to numpy array. This only works for tensors on CPU, hence first push to CPU
    # output_image = output_image.cpu().numpy()
    # plt.imshow(output_image, origin='lower', extent=(x1lower, x1upper, x2lower, x2upper), zorder = -1)
    
    plt.grid(False)
    plt.xlim([x1lower, x1upper])
    plt.ylim([x2lower, x2upper])
    # plt.axis('scaled')

    # labels_predicted = [0 if value <= 0.5 else 1 for value in labels_predicted.numpy()]
    if contour:
        colors = [to_rgb("C1"), [1, 1, 1], to_rgb("C0")] # first color is black, last is red
        cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=40)
        z = np.array(preds).reshape(xx1.shape)
        
        levels = np.linspace(0.,1.,8).tolist()
        
        cont = plt.contourf(xx1, xx2, z, levels, alpha=1, cmap=cm, zorder = 0, extent=(x1lower, x1upper, x2lower, x2upper)) #plt.get_cmap('coolwarm')
        cbar = fig.colorbar(cont, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('prediction prob.')



    # preds_contour = preds.view(len(x1), len(x1)).detach()
    # plt.contourf(xx1, xx2, preds_contour, alpha=1)
    if fig_name:
        plt.savefig(fig_name + '.png', bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
    return fig


@torch.no_grad()
def classification_levelsets(model, fig_name=None, footnote=None, contour = True, plotlim = [-2, 2]):
    
    
    x1lower, x1upper = plotlim
    x2lower, x2upper = plotlim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fig = plt.figure(figsize=(5, 5), dpi=100)
    
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
    else: plt.show()
        
def loss_evolution(trainer, epoch, filename = '', figsize = None, footnote = None):
    print(f'{epoch = }')
    fig = plt.figure(dpi = 100, figsize=(figsize))
    labelsize = 10

    #plot whole loss history in semi-transparent
    epoch_scale = range(1,len(trainer.histories['epoch_loss_history']) + 1)
    epoch_scale = list(epoch_scale)
    plt.plot(epoch_scale,trainer.histories['epoch_loss_history'], 'k', alpha = 0.5 )
    plt.plot(epoch_scale, trainer.histories['epoch_loss_rob_history'], 'C2--', zorder = -1, alpha = 0.5)
    
    if trainer.eps > 0: #if the trainer has a robustness term
        standard_loss_term = [loss - rob for loss, rob in zip(trainer.histories['epoch_loss_history'],trainer.histories['epoch_loss_rob_history'])]
        plt.plot(epoch_scale, standard_loss_term,'C1--', alpha = 0.5)
        leg = plt.legend(['total loss', 'gradient term', 'standard term'], prop= {'size': labelsize})
    else: leg = plt.legend(['standard loss', '(inactive) gradient term'], prop= {'size': labelsize})
        
    #set alpha to 1
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    plt.plot(epoch_scale[0:epoch], trainer.histories['epoch_loss_history'][0:epoch], color = 'k')
    plt.scatter(epoch, trainer.histories['epoch_loss_history'][epoch-1], color = 'k' , zorder = 1)
    
    plt.plot(epoch_scale[0:epoch], trainer.histories['epoch_loss_rob_history'][0:epoch], 'C2--')
    plt.scatter(epoch, trainer.histories['epoch_loss_rob_history'][epoch - 1], color = 'C2', zorder = 1)
    
    if trainer.eps > 0: #if the trainer has a robustness term
        plt.plot(epoch_scale[0:epoch], standard_loss_term[0:epoch],'--', color = 'C1')
        plt.scatter(epoch, standard_loss_term[epoch - 1], color = 'C1', zorder = 1)
        
    plt.xlim(1, len(trainer.histories['epoch_loss_history']))
    # plt.ylim([0,0.75])
    plt.yticks(np.arange(0,1,0.25))
    plt.grid(zorder = -2)
    # plt.tight_layout()
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.set_aspect('auto')
    ax.set_axisbelow(True)
    plt.xlabel('Epochs', size = labelsize)
    if trainer.eps > 0:
        plt.ylabel('Loss Robust', size = labelsize)
        
    else:
        plt.ylabel('Loss Standard', size = labelsize)

    if footnote:
        plt.figtext(0.5, -0.005, footnote, ha="center", fontsize=9)

    if not filename == '':
        plt.savefig(filename + '.png', bbox_inches='tight', dpi=100, format='png', facecolor = 'white')
        plt.clf()
        plt.close()
        
    else:
        plt.show()
        print('no filename given')
        

def comparison_plot(filename1, title1, filename2, title2, filename_output, figsize = None, show = False, dpi = 100):
    plt.figure(dpi = dpi, figsize=figsize)
    plt.subplot(121)
    sub1 = imageio.imread(filename1)
    plt.imshow(sub1)
    plt.title(title1)
    plt.axis('off')

    plt.subplot(122)
    sub2 = imageio.imread(filename2)
    plt.imshow(sub2)
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(filename_output, bbox_inches='tight', dpi=dpi, format='png', facecolor = 'white')
    if show: plt.show()
    else:
        plt.gca()
        plt.close()
        
        
def train_to_classifier_imgs(model, trainer, dataloader, subfolder, num_epochs, plotfreq, filename = '', plotlim = [-2, 2]):
    
    if not os.path.exists(subfolder):
            os.makedirs(subfolder)

    fig_name_base = os.path.join(subfolder,'') #os independent file path

    for epoch in range(0,num_epochs,plotfreq):
        trainer.train(dataloader, plotfreq)
        epoch_trained = epoch + plotfreq
        classification_levelsets(model, fig_name = fig_name_base + filename + str(epoch_trained), footnote = f'epoch = {epoch_trained}', plotlim = plotlim)
        print(f'\n Plot {epoch_trained =}')