#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski (adapted from https://github.com/EmilienDupont/augmented-neural-odes)
"""
import json
import torch.nn as nn
import numpy as np
from numpy import mean
import torch
# from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from torch.utils import data as data
from torch.utils.data import DataLoader, TensorDataset






losses = {'mse': nn.MSELoss(), 
          'cross_entropy': nn.CrossEntropyLoss(), 
          'ell1': nn.SmoothL1Loss()
}

class Trainer():
    """
    Given an optimizer, we write the training loop for minimizing the functional.
    We need several hyperparameters to define the different functionals.

    ***
    -- The boolean "turnpike" indicates whether we integrate the training error over [0,T]
    where T is the time horizon intrinsic to the model.
    -- The boolean "fixed_projector" indicates whether the output layer is given or trained
    -- The float "bound" indicates whether we consider L1+Linfty reg. problem (bound>0.), or 
    L2 reg. problem (bound=0.). If bound>0., then bound represents the upper threshold for the 
    weights+biases.
    ***
    """
    def __init__(self, model, optimizer, device, cross_entropy=True,
                 print_freq=10, record_freq=10, verbose=True, save_dir=None, 
                 turnpike=True, bound=0., fixed_projector=False):
        self.model = model
        self.optimizer = optimizer
        self.cross_entropy = cross_entropy
        self.device = device
        if cross_entropy:
            self.loss_func = losses['cross_entropy']
        else:
            #self.loss_func = losses['mse']
            self.loss_func = nn.MultiMarginLoss()
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.steps = 0
        self.save_dir = save_dir
        self.verbose = verbose
        self.turnpike = turnpike
        # In case we consider L1-reg. we threshold the norm. 
        # Examples: M \sim T for toy datasets; 200 for mnist
        self.threshold = bound    
        self.fixed_projector = fixed_projector

        self.histories = {'loss_history': [], 'acc_history': [],
                          'epoch_loss_history': [], 'epoch_acc_history': []}
        self.buffer = {'loss': [], 'accuracy': []}
        self.is_resnet = hasattr(self.model, 'num_layers')

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            avg_loss = self._train_epoch(data_loader, epoch)
            if self.verbose:
                print("Epoch {}: {:.3f}".format(epoch + 1, avg_loss))

    def _train_epoch(self, data_loader, epoch):
        epoch_loss = 0.
        epoch_acc = 0.
        for i, (x_batch, y_batch) in enumerate(data_loader):
            self.optimizer.zero_grad()
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            if not self.is_resnet:
                y_pred, traj = self.model(x_batch)   
                time_steps = self.model.time_steps 
                T = self.model.T
                dt = T/time_steps
            else:
                # In ResNet, dt=1=T/N_layers.
                y_pred, traj, _ = self.model(x_batch)
                time_steps = self.model.num_layers
                T = time_steps
                dt = 1 

            if not self.turnpike:                                       ## Classical empirical risk minimization
                loss = self.loss_func(y_pred, y_batch)
            else:                                                       ## Augmented empirical risk minimization
                if self.threshold>0: # l1 controls
                    l1_regularization = 0.
                    for param in self.model.parameters():
                        l1_regularization += param.abs().sum()
                    ## lambda = 5*1e-3 for spheres+inside
                    loss = 1.5*sum([self.loss_func(traj[k], y_batch)+self.loss_func(traj[k+1], y_batch) 
                                    for k in range(time_steps-1)]) + 0.005*l1_regularization #this was 0.005
                    
                else: #l2 controls
                    if self.fixed_projector: #maybe not needed
                        xd = torch.tensor([[6.0/0.8156, 0.5/(2*0.4525)] if x==1 else [-6.0/0.8156, -2.0/(2*0.4525)] for x in y_batch])
                        loss = self.loss_func(y_pred, y_batch.float())+sum([self.loss_func(traj[k], xd)
                                            +self.loss_func(traj[k+1], xd) for k in range(time_steps-1)])
                    else:
                        ## beta=1.5 for point clouds, trapizoidal rule to integrate
                        beta = 1.75                      
                        loss = beta*sum([self.loss_func(traj[k], y_batch)+self.loss_func(traj[k+1], y_batch) 
                                        for k in range(time_steps-1)])
            loss.backward()
            self.optimizer.step()
                        
            if self.cross_entropy:
                epoch_loss += self.loss_func(traj[-1], y_batch).item()   
                m = nn.Softmax()
                softpred = m(y_pred)
                softpred = torch.argmax(softpred, 1)  
                epoch_acc += (softpred == y_batch).sum().item()/(y_batch.size(0))       
            else:
                epoch_loss += self.loss_func(y_pred, y_batch).item()
        
            if i % self.print_freq == 0:
                if self.verbose:
                    print("\nEpoch {}/{}".format(i, len(data_loader)))
                    if self.cross_entropy:
                        print("Loss: {:.3f}".format(self.loss_func(traj[-1], y_batch).item()))
                        print("Accuracy: {:.3f}".format((softpred == y_batch).sum().item()/(y_batch.size(0))))
                       
                    else:
                        print("Loss: {:.3f}".format(self.loss_func(y_pred, y_batch).item()))
                        
            self.buffer['loss'].append(self.loss_func(traj[-1], y_batch).item())
            if not self.fixed_projector and self.cross_entropy:
                self.buffer['accuracy'].append((softpred == y_batch).sum().item()/(y_batch.size(0)))

            # At every record_freq iteration, record mean loss and clear buffer
            if self.steps % self.record_freq == 0:
                self.histories['loss_history'].append(mean(self.buffer['loss']))
                if not self.fixed_projector and self.cross_entropy:
                    self.histories['acc_history'].append(mean(self.buffer['accuracy']))

                # Clear buffer
                self.buffer['loss'] = []
                self.buffer['accuracy'] = []

                # Save information in directory
                if self.save_dir is not None:
                    dir, id = self.save_dir
                    with open('{}/losses{}.json'.format(dir, id), 'w') as f:
                        json.dump(self.histories['loss_history'], f)

            self.steps += 1

        # Record epoch mean information
        self.histories['epoch_loss_history'].append(epoch_loss / len(data_loader))
        if not self.fixed_projector:
            self.histories['epoch_acc_history'].append(epoch_acc / len(data_loader))

        return epoch_loss / len(data_loader)



class doublebackTrainer():
    """
    Given an optimizer, we write the training loop for minimizing the functional.
    We need several hyperparameters to define the different functionals.

    ***
    -- The boolean "turnpike" indicates whether we integrate the training error over [0,T]
    where T is the time horizon intrinsic to the model.
    -- The boolean "fixed_projector" indicates whether the output layer is given or trained
    -- The float "bound" indicates whether we consider L1+Linfty reg. problem (bound>0.), or 
    L2 reg. problem (bound=0.). If bound>0., then bound represents the upper threshold for the 
    weights+biases.
    -- eps: Set a strength for the extra loss term that penalizes the gradients of the original loss
    -- The float eps_comp records the gradient of the standard loss even when robust training is not active (for comparison). Only to be used with eps = 0
    ***
    """
    def __init__(self, model, optimizer, device, cross_entropy=True,
                 print_freq=10, record_freq=10, verbose=True, save_dir=None, 
                 turnpike=True, bound=0., fixed_projector=False, eps = 0.01, l2_factor = 0, eps_comp = 0., db_type = 'l1'):
        self.model = model
        self.optimizer = optimizer
        self.cross_entropy = cross_entropy
        self.device = device
        if cross_entropy:
            self.loss_func = losses['cross_entropy']
        else:
            #self.loss_func = losses['mse']
            self.loss_func = nn.MultiMarginLoss()
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.steps = 0
        self.save_dir = save_dir
        self.verbose = verbose
        self.turnpike = turnpike
        # In case we consider L1-reg. we threshold the norm. 
        # Examples: M \sim T for toy datasets; 200 for mnist
        self.threshold = bound    
        self.fixed_projector = fixed_projector

        self.histories = {'loss_history': [], 'loss_rob_history': [],'acc_history': [],
                          'epoch_loss_history': [], 'epoch_loss_rob_history': [],  'epoch_acc_history': []}
        self.buffer = {'loss': [], 'loss_rob': [], 'accuracy': []}
        self.is_resnet = hasattr(self.model, 'num_layers')
        self.eps = eps
        self.eps_comp = eps_comp
        self.l2_factor = l2_factor
        self.db_type = db_type
        
        # logging_dir='runs/our_experiment'
        # writer = SummaryWriter(logging_dir)

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            avg_loss = self._train_epoch(data_loader, epoch)
            if self.verbose:
                print("Epoch {}: {:.3f}".format(epoch + 1, avg_loss))

    def _train_epoch(self, data_loader, epoch):
        epoch_loss = 0.
        epoch_loss_rob = 0.
        epoch_acc = 0.

        
        #If eps = 0, we have standard training, if eps_comp is greater 0, we have standard training but record the gradient term as comparison
        #if eps > 0 we activate robust training and record the gradient term
        eps_eff = max(self.eps_comp, self.eps)
        # print(eps_eff)
        loss_max = torch.tensor(0.)


        x_batch_grad = torch.tensor(0.).to(self.device)
        
        for i, (x_batch, y_batch) in enumerate(data_loader):
                # if i == 0:
                #     print('first data batch', x_batch[0], y_batch[0])
            self.optimizer.zero_grad()
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            
            
            if eps_eff > 0.: #!!!!
                x_batch.requires_grad = True #i need this for calculating the gradient term
            
            if not self.is_resnet:
                y_pred, traj = self.model(x_batch)   
                time_steps = self.model.time_steps 
                T = self.model.T
                dt = T/time_steps
            else:
                # In ResNet, dt=1=T/N_layers.
                y_pred, traj, _ = self.model(x_batch)
                time_steps = self.model.num_layers
                T = time_steps
                dt = 1 

                                               ## Classical empirical risk minimization
            loss = self.loss_func(y_pred, y_batch)
            loss_rob = torch.tensor(0.)
            # v = torch.tensor([0,1.])
            #adding perturbed trajectories
            
            if self.l2_factor > 0:
                for param in self.model.parameters():
                    l2_regularization = param.norm()
                    loss += self.l2_factor * l2_regularization
            
            if eps_eff > 0.:
                x_batch_grad = torch.autograd.grad(loss, x_batch, create_graph=True, retain_graph=True)[0] #not sure if retrain_graph is necessary here
                
                if self.db_type == 'l1':
                    loss_rob = x_batch_grad.abs().sum() #this corresponds to linfty defense
                    
                if self.db_type == 'l2':
                    loss_rob = x_batch_grad.norm() #this corresponds to l2 defense
                    
                loss_rob = eps_eff * loss_rob
            

            

            if (self.eps > 0.) and (self.eps == eps_eff): #robust loss term is active or is logged + make sure there is no confusing between epsilon of logging and training epsilon
                loss = (1-self.eps)*loss + loss_rob
                # print(f'{loss=}')
                # loss = (1-eps) * loss + eps * adj_term #was 0.005 before
            loss.backward()
            self.optimizer.step()
        
            
            if self.cross_entropy:
                epoch_loss += loss.item()
                epoch_loss_rob += loss_rob.item() 
                m = nn.Softmax(dim = 1)
                # print(y_pred.size())
                softpred = m(y_pred)
                softpred = torch.argmax(softpred, 1)  
                epoch_acc += (softpred == y_batch).sum().item()/(y_batch.size(0))       
            else:
                epoch_loss += loss.item()
                epoch_loss_rob += loss_rob.item()
                
        
            if i % self.print_freq == 0:
                if self.verbose:
                    print("\nIteration {}/{}".format(i, len(data_loader)))
                    if self.cross_entropy:
                        print("Loss: {:.3f}".format(loss))
                        print("Robust Term Loss: {:.3f}".format(loss_rob))
                        
                        print("Accuracy: {:.3f}".format((softpred == y_batch).sum().item()/(y_batch.size(0))))
                       
                    else:
                        print("Loss: {:.3f}".format(loss))
                        
            self.buffer['loss'].append(loss.item())
            self.buffer['loss_rob'].append(loss_rob.item())
            
            
            if not self.fixed_projector and self.cross_entropy:
                self.buffer['accuracy'].append((softpred == y_batch).sum().item()/(y_batch.size(0)))

            # At every record_freq iteration, record mean loss and clear buffer
            if self.steps % self.record_freq == 0:
                self.histories['loss_history'].append(mean(self.buffer['loss']))
                self.histories['loss_rob_history'].append(mean(self.buffer['loss_rob']))
                if not self.fixed_projector and self.cross_entropy:
                    self.histories['acc_history'].append(mean(self.buffer['accuracy']))

                # Clear buffer
                self.buffer['loss'] = []
                self.buffer['loss_rob'] = []
                self.buffer['accuracy'] = []

                # Save information in directory
                if self.save_dir is not None:
                    dir, id = self.save_dir
                    with open('{}/losses{}.json'.format(dir, id), 'w') as f:
                        json.dump(self.histories['loss_history'], f)

            self.steps += 1

        # Record epoch mean information
        self.histories['epoch_loss_history'].append(epoch_loss / len(data_loader))
        self.histories['epoch_loss_rob_history'].append(epoch_loss_rob / len(data_loader))
        
        # self.histories['ep']
        if not self.fixed_projector:
            self.histories['epoch_acc_history'].append(epoch_acc / len(data_loader))

        return epoch_loss / len(data_loader)
    
    
    
    
class epsTrainer():
    """
    Given an optimizer, we write the training loop for minimizing the functional.
    We need several hyperparameters to define the different functionals.

    ***
    -- The boolean "turnpike" indicates whether we integrate the training error over [0,T]
    where T is the time horizon intrinsic to the model.
    -- The boolean "fixed_projector" indicates whether the output layer is given or trained
    -- The float "bound" indicates whether we consider L1+Linfty reg. problem (bound>0.), or 
    L2 reg. problem (bound=0.). If bound>0., then bound represents the upper threshold for the 
    weights+biases.
    ***
    """
    def __init__(self, model, optimizer, device, cross_entropy=True,
                 print_freq=10, record_freq=10, verbose=True, save_dir=None, 
                 turnpike=True, bound=0., fixed_projector=False, eps = 0.01, alpha = 0.01):
        self.model = model
        self.optimizer = optimizer
        self.cross_entropy = cross_entropy
        self.device = device
        if cross_entropy:
            self.loss_func = losses['cross_entropy']
        else:
            #self.loss_func = losses['mse']
            self.loss_func = nn.MultiMarginLoss()
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.steps = 0
        self.save_dir = save_dir
        self.verbose = verbose
        self.turnpike = turnpike
        # In case we consider L1-reg. we threshold the norm. 
        # Examples: M \sim T for toy datasets; 200 for mnist
        self.threshold = bound    
        self.fixed_projector = fixed_projector

        self.histories = {'loss_history': [], 'acc_history': [],
                          'epoch_loss_history': [], 'epoch_acc_history': []}
        self.buffer = {'loss': [], 'accuracy': []}
        self.is_resnet = hasattr(self.model, 'num_layers')
        self.eps = eps
        self.alpha = alpha #strength of robustness term

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            avg_loss = self._train_epoch(data_loader, epoch)
            if self.verbose:
                print("Epoch {}: {:.3f}".format(epoch + 1, avg_loss))

    def _train_epoch(self, data_loader, epoch):
        epoch_loss = 0.
        epoch_acc = 0.

        v_steps = 5
        v = torch.zeros(v_steps,2)
        eps = self.eps
        alpha = self.alpha
        loss_max = torch.tensor(0.)


        
        # for k in range(v_steps):
        #     t = k*(2*torch.tensor(math.pi))/v_steps
        #     v[k] = torch.tensor([torch.sin(t),torch.cos(t)])
    #generate perturbed directions
        x_batch_grad = torch.tensor(0.)
        for i, (x_batch, y_batch) in enumerate(data_loader):
                # if i == 0:
                #     print('first data batch', x_batch[0], y_batch[0])
            self.optimizer.zero_grad()
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            x_batch.requires_grad = True #i need this for Fast sign gradient method
            
            if not self.is_resnet:
                y_pred, traj = self.model(x_batch)   
                time_steps = self.model.time_steps 
                T = self.model.T
                dt = T/time_steps
            else:
                # In ResNet, dt=1=T/N_layers.
                y_pred, traj, _ = self.model(x_batch)
                time_steps = self.model.num_layers
                T = time_steps
                dt = 1 

            if not self.turnpike:                                       ## Classical empirical risk minimization
                loss = self.loss_func(y_pred, y_batch)
                # v = torch.tensor([0,1.])
                #adding perturbed trajectories
                
                if eps > 0.:
                    # loss_max = torch.tensor(0.)
                    
                    #Generate the sign gradient vector
                    # loss.backward() #previously here i had retain_graph=True. i am not sure why i thought i needed it
                    # x_batch_grad = x_batch.grad.data.sign()

                    x_batch_grad = torch.autograd.grad(loss, x_batch, create_graph=True, retain_graph=True)[0]
                    x_batch_grad = x_batch_grad.sign()
                    # print('grad size',x_batch_grad.size())
                    # print('size', x_batch.size())
                    
                    # for k in range(v_steps):
                    #     y_eps, traj_eps = self.model(x_batch + eps*v[k]) #model for perturbed input
                    #     loss_v = (traj_eps - traj).abs().sum(dim = 0) #for trapezoidal rule. endpoints not regarded atm
                    #     loss_max = torch.maximum(loss_max,loss_v)
                        # print('loss max', loss_max.sum())
                        # print('loss_v', loss_v)
                        # print('loss max',loss_max)
                    # loss += 0.005*loss_max.sum()
                    # print('loss',loss)

                   

                    
                    ###########################
                    #this should only add an extra loss to the batch items that differ from the unperturbed prediction more than pert
                    y_pred_eps, _ = self.model(x_batch + eps * x_batch_grad)
                    # y_pred, _ = self.model(x_batch)
                    
                    # pert = 0.01
                    # diff = torch.abs(y_pred_eps - y_pred)
                    # cond = diff > pert
                    # y_eff = torch.where(cond, y_pred_eps, torch.tensor(0, dtype=y_pred_eps.dtype))
                    y_eff = y_pred_eps #comment this if you use other
                    ############################

                    # print('y_eff', y_eff)
                    loss = (1-alpha) * loss + alpha * self.loss_func(y_eff, y_batch) #was 0.005 before
            else:                                                       ## Augmented empirical risk minimization
                if self.threshold>0: # l1 controls
                    l1_regularization = 0.
                    for param in self.model.parameters():
                        l1_regularization += param.abs().sum()
                    ## lambda = 5*1e-3 for spheres+inside
                    loss = 1.5*sum([self.loss_func(traj[k], y_batch)+self.loss_func(traj[k+1], y_batch) 
                                    for k in range(time_steps-1)]) + 0.005*l1_regularization #this was 0.005
                    
                else: #l2 controls
                    if self.fixed_projector: #maybe not needed
                        xd = torch.tensor([[6.0/0.8156, 0.5/(2*0.4525)] if x==1 else [-6.0/0.8156, -2.0/(2*0.4525)] for x in y_batch])
                        loss = self.loss_func(y_pred, y_batch.float())+sum([self.loss_func(traj[k], xd)
                                            +self.loss_func(traj[k+1], xd) for k in range(time_steps-1)])
                    else:
                        ## beta=1.5 for point clouds, trapizoidal rule to integrate
                        beta = 1.75                      
                        loss = beta*sum([self.loss_func(traj[k], y_batch)+self.loss_func(traj[k+1], y_batch) 
                                        for k in range(time_steps-1)])
            loss.backward()
            self.optimizer.step()
        
            if self.threshold>0: 
                self.model.apply(clipper)       # We apply the Linfty constraint to the trained parameters
            
            if self.cross_entropy:
                epoch_loss += self.loss_func(traj[-1], y_batch).item()   
                m = nn.Softmax()
                softpred = m(y_pred)
                softpred = torch.argmax(softpred, 1)  
                epoch_acc += (softpred == y_batch).sum().item()/(y_batch.size(0))       
            else:
                epoch_loss += self.loss_func(y_pred, y_batch).item()
        
            if i % self.print_freq == 0:
                if self.verbose:
                    print("\nEpoch {}/{}".format(i, len(data_loader)))
                    if self.cross_entropy:
                        print("Loss: {:.3f}".format(self.loss_func(traj[-1], y_batch).item()))
                        print("Accuracy: {:.3f}".format((softpred == y_batch).sum().item()/(y_batch.size(0))))
                       
                    else:
                        print("Loss: {:.3f}".format(self.loss_func(y_pred, y_batch).item()))
                        
            self.buffer['loss'].append(self.loss_func(traj[-1], y_batch).item())
            if not self.fixed_projector and self.cross_entropy:
                self.buffer['accuracy'].append((softpred == y_batch).sum().item()/(y_batch.size(0)))

            # At every record_freq iteration, record mean loss and clear buffer
            if self.steps % self.record_freq == 0:
                self.histories['loss_history'].append(mean(self.buffer['loss']))
                if not self.fixed_projector and self.cross_entropy:
                    self.histories['acc_history'].append(mean(self.buffer['accuracy']))

                # Clear buffer
                self.buffer['loss'] = []
                self.buffer['accuracy'] = []

                # Save information in directory
                if self.save_dir is not None:
                    dir, id = self.save_dir
                    with open('{}/losses{}.json'.format(dir, id), 'w') as f:
                        json.dump(self.histories['loss_history'], f)

            self.steps += 1

        # Record epoch mean information
        self.histories['epoch_loss_history'].append(epoch_loss / len(data_loader))
        if not self.fixed_projector:
            self.histories['epoch_acc_history'].append(epoch_acc / len(data_loader))

        return epoch_loss / len(data_loader)

    def x_grad(self, x_batch, y_batch):
        x_batch.requires_grad = True

        x_batch_grad = torch.tensor(0.)
        
        y_pred, _ = self.model(x_batch)
        loss = self.loss_func(y_pred, y_batch)

        self.optimizer.zero_grad()
        
        
        x_batch_grad = torch.autograd.grad(loss, x_batch)[0]
        x_batch.requires_grad = False
        return x_batch_grad

        
                

                                             ## Classical empirical risk minimization
                

def create_dataloader(data_type, batch_size = 3000, noise = 0.15, factor = 0.15, random_state = 1, shuffle = True, plotlim = [-2, 2]):
    if data_type == 'circles':
        X, y = make_circles(batch_size, noise=noise, factor=factor, random_state=random_state, shuffle = shuffle)
        
        
    elif data_type == 'blobs':
        centers = [[-1, -1], [1, 1]]
        X, y = make_blobs(
    n_samples=batch_size, centers=centers, cluster_std=noise, random_state=random_state)
        
        
    elif data_type == 'moons':
        X, y = make_moons(batch_size, noise = noise, shuffle = shuffle , random_state = random_state)
    
    
    elif data_type == 'xor':
        X = torch.randint(low=0, high=2, size=(batch_size, 2), dtype=torch.float32)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).float()
        # y = y.to(torch.int64)
        X += noise * torch.randn(X.shape)
        
        
    else: 
        print('datatype not supported')
        return None, None

    g = torch.Generator()
    g.manual_seed(random_state)
    
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=random_state, shuffle = shuffle)

    X_train = torch.Tensor(X_train) # transform to torch tensor for dataloader
    y_train = torch.Tensor(y_train) #transform to torch tensor for dataloader

    X_test = torch.Tensor(X_test) # transform to torch tensor for dataloader
    y_test = torch.Tensor(y_test) #transform to torch tensor for dataloader

    X_train = X_train.type(torch.float32)  #type of orginial pickle.load data
    y_train = y_train.type(torch.int64) #dtype of original picle.load data

    X_test = X_test.type(torch.float32)  #type of orginial pickle.load data
    y_test = y_test.type(torch.int64) #dtype of original picle.load data


    train_data = TensorDataset(X_train,y_train) # create your datset
    test_data = TensorDataset(X_test, y_test)

    train = DataLoader(train_data, batch_size=64, shuffle=shuffle, generator=g)
    test = DataLoader(test_data, batch_size=256, shuffle=shuffle, generator = g) #128 before
    
    data_0 = X_train[y_train == 0]
    data_1 = X_train[y_train == 1]
    fig = plt.figure(figsize = (5,5), dpi = 100)
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333",  alpha = 0.5)
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", alpha = 0.5)
    plt.xlim(plotlim)
    plt.ylim(plotlim)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig('trainingset.png', bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
    plt.show()
    
    return train, test