import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import DataLoader,TensorDataset
from plots.gifs import trajectory_gif
from plots.plots import get_feature_history, plt_train_error, plt_norm_state, plt_norm_control, plt_classifier, feature_plot, plt_dataset
from models.training import Trainer, robTrainer, epsTrainer, epslinTrainer
from models.neural_odes import NeuralODE, robNeuralODE
from models.resnets import ResNet
# import pickle
# import sys
import matplotlib.pyplot as plt
import imageio
import math

from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split

#if training = False, models will be loaded from file


torch.backends.cudnn.deterministic = True
torch.manual_seed(2)
torch.cuda.manual_seed(2)


hidden_dim, data_dim = 2, 2 
T, num_steps = 5.0, 20  #T is the end time, num_steps are the amount of discretization steps for the ODE solver
dt = T/num_steps
turnpike = False
bound = 0.
fp = False
cross_entropy = True
noise = 0.05
shuffle = False
non_linearity = 'relu' #'tanh'
architecture = 'bottleneck' #outside

# eps = 0.01
v_steps = 5


training = True #train new network or load saved one
num_epochs = 80 #number of optimization epochs for gradient decent

epsilons = [0, 0.001, 0.01,  0.1]
fig_name = '1elin_generalization_eps'





if turnpike:
    weight_decay = 0 if bound>0. else dt*0.01
else: 
    weight_decay = dt*0.01          #0.01 for fp, 0.1 else


###DATA PREPARATION
X, y = make_circles(3000, noise=noise, factor=0.15, random_state=1, shuffle = shuffle)
# X, y = make_moons(3000, noise = noise, random_state = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.05 ) #random_state=2, shuffle = shuffle)

X_train = torch.Tensor(X_train) # transform to torch tensor for dataloader
y_train = torch.Tensor(y_train) #transform to torch tensor for dataloader

X_test = torch.Tensor(X_train) # transform to torch tensor for dataloader
y_test = torch.Tensor(y_train) #transform to torch tensor for dataloader

X_train = X_train.type(torch.float32)  #type of orginial pickle.load data
y_train = y_train.type(torch.int64) #dtype of original picle.load data

X_test = X_test.type(torch.float32)  #type of orginial pickle.load data
y_test = y_test.type(torch.int64) #dtype of original picle.load data


data_line = TensorDataset(X_train,y_train) # create your datset
test = TensorDataset(X_test, y_test)



g = torch.Generator()
g.manual_seed(1)

dataloader = DataLoader(data_line, batch_size=64, shuffle=shuffle, generator=g)
dataloader_viz = DataLoader(data_line, batch_size=128, shuffle=shuffle, generator = g)



#####model initializiation and training


# anode = NeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity=non_linearity, 
#                     architecture=architecture, T=T, time_steps=num_steps, fixed_projector=fp, cross_entropy=cross_entropy)
# optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3, weight_decay=weight_decay) #weight decay parameter modifies norm
# trainer_anode = Trainer(anode, optimizer_anode, device, cross_entropy=cross_entropy, 
#                         turnpike=turnpike, bound=bound, fixed_projector=fp, verbose = False)       

# trainer_anode.train(dataloader, num_epochs)
# plt_classifier(anode, data_line, test, num_steps=10, save_fig = '1ageneralization.png') 

# print('1generalization.png created')
# #robust training




for eps in epsilons:

    torch.manual_seed(2)
    
    eps_node = NeuralODE(device, data_dim, hidden_dim, adjoint = False, augment_dim=0, non_linearity=non_linearity, 
                                architecture=architecture, T=T, time_steps=num_steps, fixed_projector=fp, cross_entropy=cross_entropy)
        
    # for name, param in eps_node.named_parameters():
    #     i = 0
    #     if param.requires_grad and i<2:
    #         print(name, param.data)
    #         i += 1

    optimizer_node = torch.optim.Adam(eps_node.parameters(), lr=1e-3, weight_decay = weight_decay) #weight decay parameter modifies norm
    trainer_eps_node = epslinTrainer(eps_node, optimizer_node, device, cross_entropy = cross_entropy, 
                            turnpike=turnpike, bound=bound, fixed_projector=fp, verbose = False, eps =  eps)
    trainer_eps_node.train(dataloader, num_epochs)
    
    footnote = 'eps = {}, epochs = {}, data_noise = {}'.format(eps, num_epochs, noise)
    plt_classifier(eps_node, data_line, test, num_steps=10, footnote = footnote, save_fig = '{}{}'.format(fig_name, eps) +'.png') 
    print('{}{} created'.format(fig_name,eps))


             






if training:
    # torch.save(anode.state_dict(), 'anode.pth')
    torch.save(eps_node.state_dict(), 'rob_node.pth')
else:
    anode.load_state_dict(torch.load('anode.pth'))
    eps_node.load_state_dict(torch.load('rob_node.pth'))


