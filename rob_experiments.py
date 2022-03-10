import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import DataLoader,TensorDataset
from plots.gifs import trajectory_gif
from plots.plots import get_feature_history, plt_train_error, plt_norm_state, plt_norm_control, plt_classifier, feature_plot, plt_dataset
from models.training import Trainer, robTrainer, epsTrainer
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
torch.manual_seed(1)
torch.cuda.manual_seed(1)


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
num_epochs = 100 #number of optimization epochs for gradient decent
rob_factor = [0.00001, 0.0001, 0.001, 0.002, 0.005]

fig_name = '1generalization_rob'





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
g.manual_seed(0)
torch.manual_seed(1)

dataloader = DataLoader(data_line, batch_size=64, shuffle=shuffle, generator=g)
dataloader_viz = DataLoader(data_line, batch_size=128, shuffle=shuffle, generator = g)



#####model initializiation and training


anode = NeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity=non_linearity, 
                    architecture=architecture, T=T, time_steps=num_steps, fixed_projector=fp, cross_entropy=cross_entropy)
optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3, weight_decay=weight_decay) #weight decay parameter modifies norm
trainer_anode = Trainer(anode, optimizer_anode, device, cross_entropy=cross_entropy, 
                        turnpike=turnpike, bound=bound, fixed_projector=fp, verbose = True)       

trainer_anode.train(dataloader, num_epochs)
footnote = 'standard trainer, epochs = {}, data_noise = {}'.format( num_epochs, noise)
plt_classifier(anode, data_line, test, num_steps=10, save_fig = '1generalization.png', footnote = footnote) 


#robust training



for factor in rob_factor:
        

    rob_node = robNeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='tanh', 
                                architecture='outside', T=T, time_steps=num_steps, fixed_projector=fp, cross_entropy=cross_entropy)


    optimizer_rob_node = torch.optim.Adam(rob_node.parameters(), lr=1e-3, weight_decay=weight_decay) #weight decay parameter modifies norm
    trainer_rob_node = robTrainer(rob_node, optimizer_rob_node, device, cross_entropy=cross_entropy, 
                            turnpike=turnpike, bound=bound, fixed_projector=fp, verbose = False, rob_factor = factor)
    print('checkpoint 1')
    trainer_rob_node.train(dataloader, num_epochs)

    footnote = 'adj trainer, epochs = {}, rob_factor = {}, data_noise = {}'.format( num_epochs, noise, factor)
    plt_classifier(rob_node, data_line, test, num_steps=10, footnote = footnote, save_fig = '{}{}'.format(fig_name,factor) +'.png') 
    print('{}{} created'.format(fig_name,factor))


             






if training:
    torch.save(anode.state_dict(), 'anode.pth')
    torch.save(rob_node.state_dict(), 'rob_node.pth')
else:
    anode.load_state_dict(torch.load('anode.pth'))
    rob_node.load_state_dict(torch.load('rob_node.pth'))


