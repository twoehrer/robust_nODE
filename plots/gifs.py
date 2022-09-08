#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tobias woehrer (adapted from borjangeshkovski)
"""
##------------#
import imageio
import torch
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import rc

from plots.plots import loss_evolution, comparison_plot
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgb


'''
Creates gif of existing level_set images and creates fittig loss plots to combine both in a subplot

subfolder: collects images from subfolder to create a gif
num_epochs, plotfreq: Has to fit together with images in subfolder
gif_name: filename in subfolder
'''
def evo_gif(trainer, num_epochs, plotfreq, subfolder, filename, title_left = 'levelsets', title_right = 'loss', fps = 1, keep = []):
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    
    imgs = []
    filename = os.path.join(subfolder, filename)
    for epoch in range(0,num_epochs,plotfreq):
        epoch_trained = epoch + plotfreq
        print(epoch_trained)
        loss_evolution(trainer, epoch_trained, 'loss_pic')
        fig1_name = filename + str(epoch_trained) + '.png'
        comp_fig_running = filename + '_comp' + '.png'
        comparison_plot(fig1_name, title_left, 'loss_pic.png', title_right , comp_fig_running, show = False, figsize = (10,5))
        imgs.append(imageio.imread(comp_fig_running))
        if epoch_trained in keep:
            os.replace(comp_fig_running, filename + str(epoch_trained) + '_comp' + '.png')
            # if not (num_steps == layer + 1): os.remove(fig_name_rob) #keep last image
    gif_name = filename + 'evo.gif'
    imageio.mimwrite(gif_name, imgs, fps = fps)
    print(gif_name, ' created')
    
    return gif_name


def trajectory_gif(model, inputs, targets, timesteps, dpi=200, alpha=0.9,
                   alpha_line=1, filename='trajectory.gif', axlim = 0):
    
    from matplotlib import rc
    from scipy.interpolate import interp1d
    rc("text", usetex = False)
    font = {'size'   : 18}
    rc('font', **font)

    if not filename.endswith(".gif"):
        raise RuntimeError("Name must end in with .gif, but ends with {}".format(filename))
    base_filename = filename[:-4]

    ## We focus on 3 colors at most
    if False in (t < 2 for t in targets): 
        color = ['mediumpurple' if targets[i] == 2.0 else 'gold' if targets[i] == 0.0 else 'mediumseagreen' for i in range(len(targets))]
    else:
        #color = ['crimson' if targets[i, 0] > 0.0 else 'dodgerblue' for i in range(len(targets))]
        color = ['C1' if targets[i] > 0.0 else 'C0' for i in range(len(targets))]

    trajectories = model.flow.trajectory(inputs, timesteps).detach()
    num_dims = trajectories.shape[2]

    if axlim == 0:        
        x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
        y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
    else: 
        x_min, x_max = -axlim, axlim  #to normalize for rob and standard nODE
        y_min, y_max = -axlim, axlim   #
        
    if num_dims == 3:
        z_min, z_max = trajectories[:, :, 2].min(), trajectories[:, :, 2].max()
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range
    if num_dims == 3:
        z_range = z_max - z_min
        z_min -= margin * z_range
        z_max += margin * z_range
        
    T = model.T 
    integration_time = torch.linspace(0.0, T, timesteps)
    
    interp_x = []
    interp_y = []
    interp_z = []
    for i in range(inputs.shape[0]):
        interp_x.append(interp1d(integration_time, trajectories[:, i, 0], kind='cubic', fill_value='extrapolate'))
        interp_y.append(interp1d(integration_time, trajectories[:, i, 1], kind='cubic', fill_value='extrapolate'))
        if num_dims == 3:
            interp_z.append(interp1d(integration_time, trajectories[:, i, 2], kind='cubic', fill_value='extrapolate'))
    
    interp_time = 40
    # interp_time = 3 #this was 20 before
    _time = torch.linspace(0., T, interp_time)

    plt.rc('grid', linestyle="dotted", color='lightgray')
    for t in range(interp_time):
        if num_dims == 2:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            label_size = 13
            plt.rcParams['xtick.labelsize'] = label_size
            plt.rcParams['ytick.labelsize'] = label_size 
            ax.set_axisbelow(True)
            ax.xaxis.grid(color='lightgray', linestyle='dotted')
            ax.yaxis.grid(color='lightgray', linestyle='dotted')
            ax.set_facecolor('whitesmoke')
            
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.rc('text', usetex=False)
            plt.rc('font', family='serif')
            plt.xlabel(r'$x_1$', fontsize=12)
            plt.ylabel(r'$x_2$', fontsize=12)
            plt.scatter([x(_time)[t] for x in interp_x], 
                         [y(_time)[t] for y in interp_y], 
                         c=color, alpha=alpha, marker = 'o', linewidth=0.65, edgecolors='black', zorder=3)

            if t > 0:
                for i in range(inputs.shape[0]):
                    x_traj = interp_x[i](_time)[:t+1]
                    y_traj = interp_y[i](_time)[:t+1]
                    plt.plot(x_traj, y_traj, c=color[i], alpha=alpha_line, linewidth = 0.75, zorder=1)
            
        
            
        elif num_dims == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            label_size = 12
            plt.rcParams['xtick.labelsize'] = label_size
            plt.rcParams['ytick.labelsize'] = label_size 
            
            ax.scatter([x(_time)[t] for x in interp_x], 
                        [y(_time)[t] for y in interp_y],
                        [z(_time)[t] for z in interp_z],
                        c=color, alpha=alpha, marker = 'o', linewidth=0.65, edgecolors='black')
            plt.rc('text', usetex=False)
            plt.rc('font', family='serif')           
            if t > 0:
                for i in range(inputs.shape[0]):
                    x_traj = interp_x[i](_time)[:t+1]
                    y_traj = interp_y[i](_time)[:t+1]
                    z_traj = interp_z[i](_time)[:t+1]
                    ax.plot(x_traj, y_traj, z_traj, c=color[i], alpha=alpha_line, linewidth = 0.75)

            ax.set_xlim3d(x_min, x_max)
            ax.set_ylim3d(y_min, y_max)
            ax.set_zlim3d(z_min, z_max)
            
            plt.rc('grid', linestyle="dotted", color='lightgray')
            ax.grid(True)
            plt.locator_params(nbins=4)
        ax.set_aspect('equal')

        plt.savefig(base_filename + "{}.png".format(t),
                    format='png', dpi=dpi, bbox_inches='tight', facecolor = 'white')
        # Save only 3 frames (.pdf for paper)
        # if t in [0, interp_time//5, interp_time//2, interp_time-1]:
        #     plt.savefig(base_filename + "{}.pdf".format(t), format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

    imgs = []
    for i in range(interp_time):
        img_file = base_filename + "{}.png".format(i)
        imgs.append(imageio.imread(img_file))
        if i not in [0, interp_time//5, interp_time//2, interp_time-1]: os.remove(img_file) 
    imageio.mimwrite(filename, imgs, fps = 2)
    
    

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))