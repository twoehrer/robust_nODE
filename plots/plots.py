#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""
##------------#
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interp1d
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D

def plt_state_component(model, inputs, targets, timesteps, component, save_fig='component.pdf'):
    
    rc("text", usetex = True)
    font = {'size'   : 18}
    rc('font', **font)
    alpha = 0.75

    if hasattr(model, 'num_layers'):
        import sys
        sys.exit("A faire! ResNet (MNIST)") 
    else:
        T = model.T
        if False in (t < 2 for t in targets): 
            color = ['crimson' if targets[i, 0] == 0.0 else 'dodgerblue' if targets[i,0] == 1.0 else 'green' for i in range(len(targets))]
        else: 
            color = ['crimson' if targets[i, 0] > 0.0 else 'dodgerblue' for i in range(len(targets))]
        trajectories = model.flow.trajectory(inputs, timesteps).detach()
    
    inputs_aug = inputs
    
    for i in range(inputs_aug.shape[0]):
        if hasattr(model, 'num_layers'):                                        # ResNet
            y_traj = [x[i][component].detach().numpy() for x in trajectories]
        else: 
            trajectory = trajectories[:, i, :]
            y_traj = trajectory[:, component].numpy()

        ax = plt.gca()
        ax.set_facecolor('whitesmoke')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.title(r'Component of $\mathbf{x}_{i}(t)$', fontsize=12)
        plt.xlabel(r'$t$ (layers)')
        integration_time = torch.linspace(0., T, time_steps)
        plt.plot(integration_time, y_traj, c=color[i], alpha=alpha, linewidth=0.75)
        ax.set_xlim([0, T])
        plt.rc('grid', linestyle="dotted", color='lightgray')
        ax.grid(True)
    
    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

def plt_train_error(model, inputs, targets, timesteps, save_fig='train_error.pdf'):
    """
    Plot the training error over each layer / time (not epoch!)
    """
    
    rc("text", usetex = True)
    font = {'size'   : 13}
    rc('font', **font)
    alpha = 0.9

    if hasattr(model, 'num_layers'):                                    # ResNet
        ends, _, traj = model(inputs)
        traj = np.asarray(traj)
        _ = np.asarray(_)                                               #traj -> (40, 256, 784)
        T = model.num_layers
    else:
        ends, _ = model(inputs)
        _ = _.detach()
        T = model.T
    integration_time = torch.linspace(0., T, timesteps)
    
    if model.cross_entropy:
        loss = nn.CrossEntropyLoss()                                 
        error = [loss(_[k], targets) for k in range(timesteps)]
    else:
        loss = nn.MSELoss()
        non_linearity = nn.Tanh()
        import pickle
        with open('text.txt', 'rb') as fp:
            projector = pickle.load(fp)
        error = [loss(non_linearity(_[k].matmul(projector[-2].t())+projector[-1]), targets) for k in range(timesteps)]

    # Interpolate to increase smoothness
    f2 = interp1d(integration_time, error, kind='cubic', fill_value="extrapolate")
    _time = torch.linspace(0., T, 1000)

    ax = plt.gca()
    ax.set_facecolor('whitesmoke')
    ax.set_axisbelow(True)                              #Axis beneath data
    ax.xaxis.grid(color='lightgray', linestyle='dotted')
    ax.yaxis.grid(color='lightgray', linestyle='dotted')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title(r'Decay of training error', fontsize=13)
    plt.xlabel(r'$t$ (layers)')
    
    # The training error
    plt.plot(_time, f2(_time), c='tab:red', alpha=alpha, linewidth=2.25, label=r'$\mathcal{E}(\mathbf{x}(t))$')
    ax.legend(prop={'size':10}, loc="upper right", frameon=True)
    ax.set_xlim([0, int(T)])

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()  

def plt_norm_state(model, inputs, timesteps, save_fig='norm_state.pdf'):
    """
    Plot the norm of the state trajectory x(t) and the projection Px(t) over time/layer t.
    """

    rc("text", usetex = True)
    font = {'size'   : 13}
    rc('font', **font)
    alpha = 0.9    

    if not hasattr(model, 'num_layers'):
        trajectories = model.flow.trajectory(inputs, timesteps).detach()
        ends, _ = model(inputs)
        x_norm = [np.linalg.norm(trajectories[k, :, :], ord = 'fro') for k in range(timesteps)]
        _ = _.detach()
        T = model.T
        if model.cross_entropy:
            _norm = [np.linalg.norm(_[k, :], ord = 'fro') for k in range(timesteps)]
        else:
            non_linearity = nn.Tanh()
            import pickle
            with open('text.txt', 'rb') as fp:
                projector = pickle.load(fp)
            _norm = [np.linalg.norm(non_linearity(trajectories[k,:, :].matmul(projector[-2].t())+projector[-1]), ord='fro') 
                    for k in range(timesteps)]
           #_norm = [np.linalg.norm(non_linearity(_[k, :].matmul(projector[-2].t())+projector[-1]), ord='fro') 
           #         for k in range(timesteps)]
    else:                                                                       # ResNet
        ends, _, traj = model(inputs)
        traj = np.asarray(traj)
        _ = np.asarray(_)                                                       #traj -> (40, 256, 784)
        x_norm = [torch.norm(traj[k]) for k in range(timesteps)]
        _norm = [torch.norm(_[k]) for k in range(timesteps)]
        T = model.num_layers
    
    integration_time = torch.linspace(0., T, timesteps)
    # Interpolate to increase smoothness
    f1 = interp1d(integration_time, x_norm, kind='quadratic', fill_value="extrapolate")
    f2 = interp1d(integration_time, _norm, kind='quadratic', fill_value="extrapolate")
    _time = torch.linspace(0., T, 50000)

    ax = plt.gca()
    ax.set_facecolor('whitesmoke')
    ax.set_axisbelow(True)                              #Axis beneath data
    ax.xaxis.grid(color='lightgray', linestyle='dotted')
    ax.yaxis.grid(color='lightgray', linestyle='dotted')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title('Stability of norms', fontsize=13)
    plt.xlabel(r'$t$ (layers)')
    plt.plot(_time, f1(_time), c='tab:purple', alpha=alpha, linewidth=2.25, label=r'$|\mathbf{x}(t)|^2$')
    plt.plot(_time, f2(_time), c='tab:orange', alpha=alpha, linewidth=2.25, label=r'$|P\mathbf{x}(t)|^2$')
    ax.legend(prop={'size':10}, loc="best", frameon=True)
    ax.set_xlim([0, T])

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()  

def plt_norm_control(model):
    """
    Plot the norm of the control parameters u(t) over time/layer t.
    """
    rc("text", usetex = True)
    font = {'size'   : 13}
    rc('font', **font)
    alpha = 0.9

    import pickle
    dump = []
    with open('plots/controls.txt', 'rb') as fp:
        dump = pickle.load(fp)

    simple = True
    if hasattr(model, 'num_layers'):
        raise ValueError('ResNets do not have this feature yet!')
    else:
        T, time_steps = model.T, model.time_steps
    integration_time = torch.linspace(0., T, time_steps)
    if simple:
        # For now, only works for neural ODEs; sigma inside/outside.
        w_norm = [dump[k].abs().sum() for k in range(0, 2*time_steps, 2)]
        b_norm = [dump[k].abs().sum() for k in range(0, 2*time_steps) if k%2==1]
        ctrl_norm = [x+y for x,y in zip(w_norm, b_norm)]
    else:
        w1_norm = [dump[k].abs().sum() for k in range(0, 4*time_steps) if k%4==0]
        b1_norm = [dump[k].abs().sum() for k in range(0, 4*time_steps) if k%4==1]
        w2_norm = [dump[k].abs().sum() for k in range(0, 4*time_steps) if k%4==2]
        b2_norm = [dump[k].abs().sum() for k in range(0, 4*time_steps) if k%4==3]
        w_norm = [x+y for x,y in zip(w1_norm, w2_norm)]
        b_norm = [x+y for x,y in zip(b1_norm, b2_norm)]
        ctrl_norm = [x+y for x,y in zip(w_norm, b_norm)]

    f1 = interp1d(integration_time, ctrl_norm, kind='next', fill_value="extrapolate")
    _time = torch.linspace(0., T, 150)
    
    ax = plt.gca()
    ax.set_facecolor('whitesmoke')
    ax.set_axisbelow(True)                              #Axis beneath data
    ax.xaxis.grid(color='lightgray', linestyle='dotted')
    ax.yaxis.grid(color='lightgray', linestyle='dotted')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title(r'Parameter sparsity: $M=8$', fontsize=13)
    plt.xlabel(r'$t$ (layers)')
    plt.plot(_time, f1(_time), c='tab:blue', alpha=alpha, linewidth=2.25, label=r'$|u(t)|$')
    ax.legend(prop={'size':10}, loc="upper right", frameon=True)
    ax.set_xlim([0, T])

    save_fig = 'controls.pdf'
    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close() 

def feature_plot(feature_history, targets, alpha=0.9, filename='features.pdf'):
    """
    Plots the output features before projection to label space.
    Only works in 2d and 3d.
    """
    
    base_filename = filename[:-4]

    ## We focus on 3 colors at most
    if False in (t < 2 for t in targets): 
        color = ['mediumpurple' if targets[i] == 2.0 else 'gold' if targets[i] == 0.0 else 'mediumseagreen' for i in range(len(targets))]
    else:
        color = ['crimson' if targets[i] > 0.0 else 'dodgerblue' for i in range(len(targets))]
    
    num_dims = feature_history[0].shape[1]
    features = feature_history[-1]
    i = len(feature_history)
    
    if num_dims == 2:
        ax = plt.gca()
        ax.set_facecolor('whitesmoke')                      #Gray background
        ax.set_axisbelow(True)                              #Axis beneath data
        ax.xaxis.grid(color='lightgray', linestyle='dotted')
        ax.yaxis.grid(color='lightgray', linestyle='dotted')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        label_size = 12
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size 
        plt.xlabel(r'$x_1$', fontsize=13)
        plt.ylabel(r'$x_2$', fontsize=13)
        plt.scatter(features[:, 0].numpy(), features[:, 1].numpy(), c=color,
                    alpha=alpha, marker = 'o', linewidth=0.65, edgecolors='black')
    elif num_dims == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        label_size = 12
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size
        ax.scatter(features[:, 0].numpy(), features[:, 1].numpy(), features[:, 2].numpy(),
                   c=color, alpha=alpha, marker = 'o', linewidth=0.65, edgecolors='black')
        plt.rc('grid', linestyle="dotted", color='lightgray')
        ax.grid(True)
        plt.locator_params(nbins=4)
        
    plt.savefig(base_filename + "{}.pdf".format(i), format='pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

def plt_dataset(inputs, targets, plot_range=(-2.0, 2.0), save_fig="dataset.pdf"):

    import matplotlib as mpl
    from matplotlib import rc
    import seaborn as sns
    from torch.utils.data import DataLoader
    import pickle
    rc("text", usetex = True)
    font = {'size'   : 13}
    rc('font', **font)

    with open('data.txt', 'rb') as fp:
        data_line, test = pickle.load(fp)
    dataloader_viz = DataLoader(data_line, batch_size=800, shuffle=True)
    for inputs, targets in dataloader_viz:
        break  
    if False in (t < 2 for t in targets): 
        plot_range = (-2.5, 2.5)
        color = ['mediumpurple' if targets[i] == 2.0 else 'gold' if targets[i] == 0.0 else 'mediumseagreen' for i in range(len(targets))]
    else:
        color = ['crimson' if targets[i] > 0.0 else 'dodgerblue' for i in range(len(targets))]  
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    label_size = 13
    plt.rcParams['xtick.labelsize'] = label_size
    plt.rcParams['ytick.labelsize'] = label_size 
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='lightgray', linestyle='dotted')
    ax.yaxis.grid(color='lightgray', linestyle='dotted')
    ax.set_facecolor('whitesmoke')
            
    plt.xlim(plot_range[0], plot_range[1])
    plt.ylim(plot_range[0], plot_range[1])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel(r'$x_1$', fontsize=12)
    plt.ylabel(r'$x_2$', fontsize=12)
    plt.scatter(inputs[:,0], inputs[:,1], c=color, alpha=0.95, marker = 'o', linewidth=0.45, edgecolors='black', label='train') 
    
    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()  

def plt_classifier(model, data_line, test, plot_range=(-2.0, 2.0), num_steps=201, footnote = None, save_fig='generalization.pdf', trainer=None):
    """
    Plots the final classifier; train and test data are superposed.
    Only for toy cloud data.
    """

    import matplotlib as mpl
    from matplotlib import rc
    import seaborn as sns
    from torch.utils.data import DataLoader
    import pickle
    rc("text", usetex = False)
    font = {'size'   : 13}
    rc('font', **font)

    # with open('data.txt', 'rb') as fp:
    #     data_line, test = pickle.load(fp)


    
    dataloader_viz = DataLoader(data_line, batch_size=400, shuffle=False) #was 200
    test_viz = DataLoader(test, batch_size = 80, shuffle=False) #was 80
    for inputs, targets in dataloader_viz:
        break    
    for test_inputs, test_targets in test_viz:
        break
    if trainer:
        inputs_grad = trainer.x_grad(inputs, targets)
    # inputs_grad = inputs_grad / inputs_grad.norm(dim=1).unsqueeze(dim=1)
        inputs_grad =  100*inputs_grad #rescale for picture and squeeze in right form

    # print(f'{inputs_grad = }')
    # print(f'{inputs = }')
    
    if False in (t < 2 for t in targets): 
        plot_range = (-2.5, 2.5)
        color = ['mediumpurple' if targets[i] == 2.0 else 'gold' if targets[i] == 0.0 else 'mediumseagreen' for i in range(len(targets))]
        test_color = ['mediumpurple' if test_targets[i] == 2.0 else 'gold' if test_targets[i] == 0.0 else 'mediumseagreen' for i in range(len(test_targets))]
        cmap = mpl.cm.get_cmap("viridis_r")
        bounds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    else:
        color = ['crimson' if targets[i] > 0.0 else 'dodgerblue' for i in range(len(targets))]
        test_color = ['crimson' if test_targets[i] > 0.0 else 'dodgerblue' for i in range(len(test_targets))]
        cmap = sns.diverging_palette(250, 10, s=50, l=30, n=9, center="light", as_cmap=True)
        if model.cross_entropy:
            bounds = [0.0, 0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1.0]          # cross-entropy labels
        else: 
            print('Not cross entropy')
            bounds = [-1.0, -0.75,-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]       # mse labels
    
    grid = torch.zeros((num_steps * num_steps, 2))
    idx = 0
    for x1 in np.linspace(plot_range[0], plot_range[1], num_steps):
        for x2 in np.linspace(plot_range[0], plot_range[1], num_steps):
            grid[idx, :] = torch.Tensor([x1, x2])
            idx += 1

    if not model.cross_entropy:
        predictions, traj = model(grid)
        vmin, vmax = -1.05, 1.05
    else:
        pre_, traj = model(grid)
        m = nn.Softmax()
        predictions = m(pre_)
        predictions = torch.argmax(predictions, 1)
        vmin = 0.0
        vmax = 2.05 if False in (t < 2 for t in targets) else 1.05 
    
    pred_grid = predictions.view(num_steps, num_steps).detach()
    
    _x = np.linspace(plot_range[0], plot_range[1], num_steps)
    _y = np.linspace(plot_range[0], plot_range[1], num_steps)
        
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig = plt.figure()

    X_new, Y_new = np.meshgrid(_x,_y)
    i = plt.contourf(X_new, Y_new, pred_grid, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, alpha=1)

    cb = fig.colorbar(i)
    cb.ax.tick_params(size=0)
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False,
                        labelleft=False)
    
    plt.scatter(inputs[:,0], inputs[:,1], c=color, alpha=0.95, marker = 'o', linewidth=0.45, edgecolors='black', label='train')
    
    if trainer:
        for i in range(len(inputs[:,0])):
            plt.arrow(inputs[i, 0], inputs[i, 1], inputs_grad[i, 0], inputs_grad[i, 1], head_width=0.05, head_length=0.1, fc='k', ec='k', alpha = 0.5)
    
    # plt.scatter(test_inputs[:,0], test_inputs[:, 1], c=test_color, alpha=0.95, marker='o', linewidth=1.75, edgecolors='black', label='test')
    fig.patch.set_facecolor('white')
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', mew=0.45, mec='black', label='train',
                          markerfacecolor='lightgray', markersize=7),
                        Line2D([0], [0], marker='o', color='w', mew=1.75, mec='black', label='test',
                          markerfacecolor='lightgray', markersize=7)]

    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(-0.315,1.025), frameon=False)
    plt.title('Generalization outside training data', fontsize=13)
    plt.xlabel(r'$x_1$', fontsize=13)
    plt.ylabel(r'$x_2$', fontsize=13)
    plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=10)

    if len(save_fig):
        plt.savefig(save_fig, bbox_inches='tight', dpi = 300) #format='png'
        # plt.show()
        plt.clf()
        plt.close()




def get_feature_history(trainer, dataloader, inputs, targets, num_epochs):
    
    feature_history = []
    features, _ = trainer.model(inputs, return_features=True)
    feature_history.append(features.detach())

    for i in range(num_epochs):
        trainer.train(dataloader, 1)
        features, _ = trainer.model(inputs, return_features=True)
        feature_history.append(features.detach())
    return feature_history

def histories_plt(all_history_info, plot_type='loss', shaded_err=False,
                  labels=[], include_mean=True, 
                  time_per_epoch=[], save_fig=''):

    rc("text", usetex = False)
    font = {'size'   : 13}
    rc('font', **font)

    for i, history_info in enumerate(all_history_info):
        
        color = 'tab:pink'
        color_val = 'tab:blue'
        if plot_type == 'loss':
            histories = history_info["epoch_loss_history"]
            histories_val = history_info["epoch_loss_val_history"]
        elif plot_type == 'acc':
            histories = history_info["epoch_acc_history"]
            histories_val = history_info["epoch_acc_val_history"]

        if len(time_per_epoch):
            xlabel = "Time (seconds)"
        else:
            xlabel = "Epochs"

        if include_mean:
            ax = plt.gca()
            ax.set_facecolor('whitesmoke')
            ax.set_axisbelow(True)                              #Axis beneath data
            ax.xaxis.grid(color='lightgray', linestyle='dotted')
            ax.yaxis.grid(color='lightgray', linestyle='dotted')
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
                
            mean_history = np.array(histories).mean(axis=0)
            mean_history_val = np.array(histories_val).mean(axis=0)
            if len(time_per_epoch):
                epochs = time_per_epoch[i] * np.arange(len(histories[0]))
            else:
                epochs = list(range(len(histories[0])))

            if shaded_err:
                std_history = np.array(histories).std(axis=0)
                std_history_val = np.array(histories_val).std(axis=0)
                plt.fill_between(epochs, mean_history - std_history,
                                    mean_history + std_history, facecolor=color,
                                    alpha=0.5)
                plt.fill_between(epochs, mean_history_val - std_history_val,
                                    mean_history_val + std_history_val, facecolor=color_val,
                                    alpha=0.5)
                
            else:
                for k in range(len(histories)):
                    plt.plot(epochs, histories[k], c=color, alpha=0.1)
                    plt.plot(epochs, histories_val[k], c=color_val, alpha=0.1)

            plt.plot(epochs, mean_history, c=color, label="Train")
            plt.plot(epochs, mean_history_val, c=color_val, label="Test")
            ax.legend(prop={'size': 10}, loc="lower left", frameon=True)
            ax.set_xlim([0, len(epochs)-1])
            #locs = epochs
            #labels = range(1, len(epochs)+1)
            #plt.xticks(locs, labels)
            plt.xticks(range(0, len(epochs), max(len(epochs)//10,1)), range(1, len(epochs)+1, max(len(epochs)//10,1)))
        else:
            for k in range(len(histories)):
                plt.plot(histories[k], c=color, alpha=0.1)     
                plt.plot(histories_val[k], c=color_val, alpha=0.1) 
    
    plt.xlabel(xlabel)

    mnist = True
    if plot_type == "acc" and mnist:
        plt.ylim((0.75, 1.0))
        plt.title('Accuracy')
    else:
        plt.title('Error')

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()
