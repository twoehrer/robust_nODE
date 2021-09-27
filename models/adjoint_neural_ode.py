#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski (adapted from https://github.com/EmilienDupont/augmented-neural-odes)
"""
##------------#
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
MAX_NUM_STEPS = 1000

# Useful dicos:
activations = {'tanh_prime': 1-nn.Tanh()**2 }

#dummy x, needs to be replaced with trajectory of solution of dot(x) = f(u(t),x(t))


x_blocks = [nn.Linear(2, 2) for _ in range(10)]
dummy_x = nn.Sequential(*x_blocks)

class adj_Dynamics(nn.Module):
    """
    Structure of the nonlinear, right hand side $f(u(t), x(t)) of the neural ODE.
    We distinguish the different structures defined in the dictionary "architectures" just above.
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0, 
                non_linearity='tanh_prime', T=10, time_steps=10):
        super(Dynamics, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim

   
        self.non_linearity = activations[non_linearity]
    
        self.T = T
        self.time_steps = time_steps
        
       
        blocks1 = [nn.Linear(self.input_dim, hidden_dim) for _ in range(self.time_steps)]
        self.fc1_time = nn.Sequential(*blocks1)
            
    def forward(self, t, x=dummy_x):
        """
        The output of the class -> f(x(t), u(t)).
        """
        dt = self.T/self.time_steps
        k = int(t/dt)

        
        w_t = self.fc1_time[k].weight
        b_t = self.fc1_time[k].bias
        
        out = x.matmul(w_t.t())+b_t
        out = self.non_linearity(out)

                                                        # w1(t)\sigma(w2(t)x(t)+b2(t))+b1(t)
        # w1_t = self.fc1_time[k].weight
        # b1_t = self.fc1_time[k].bias
        # w2_t = self.fc3_time[k].weight
        # b2_t = self.fc3_time[k].bias
        #out = self.non_linearity(x.matmul(w1_t.t()) + b1_t)
        #out = out.matmul(w2_t.t()) + b2_t
        
        #Domenec Test
        out1 = torch.sqrt(self.non_linearity(x.matmul(w1_t.t())+torch.ones(self.hidden_dim) + b1_t) + 1e-6*torch.ones(self.hidden_dim))-torch.sqrt(1e-6*torch.ones(self.hidden_dim))
        out2 = torch.sqrt(self.non_linearity(-x.matmul(w1_t.t())+torch.ones(self.hidden_dim) + b1_t) + 1e-6*torch.ones(self.hidden_dim))-torch.sqrt(1e-6*torch.ones(self.hidden_dim))
        out = torch.min(out1, out2)
        out = out.matmul(w2_t.t())
        return out


class Semiflow(nn.Module):  #this is what matters
    """
    Given the dynamics f, generate the semiflow by solving x'(t) = f(u(t), x(t)).
    We concentrate on the forward Euler method - the user may change this by using
    other methods from torchdiffeq in the modules odeint and odeint_adjoint.

    ***
    - dynamics denotes the instance of the class Dynamics, defining the dynamics f(x,u)
    ***
    """
    def __init__(self, device, dynamics, tol=1e-3, adjoint=False, T=10, time_steps=10):
        super(Semiflow, self).__init__()
        self.adjoint = adjoint #here there is already an implementation of adjoint for backwards propagation
        self.device = device
        self.dynamics = dynamics
        self.tol = tol
        self.T = T
        self.time_steps = time_steps
        
    def forward(self, x, eval_times=None):
    
        dt = self.T/self.time_steps

        if eval_times is None:
            integration_time = torch.tensor([0, self.T]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        if self.dynamics.augment_dim > 0:
            x = x.view(x.size(0), -1)
            aug = torch.zeros(x.shape[0], self.dynamics.augment_dim).to(self.device)
            x_aug = torch.cat([x, aug], 1)
        else:
            x_aug = x

        if self.adjoint:  #maybe here we can compute both at same time, adjoint and normal
            out = odeint_adjoint(self.dynamics, x_aug, integration_time, method='euler', options={'step_size': dt})
        else:
            out = odeint(self.dynamics, x_aug, integration_time, method='euler', options={'step_size': dt})
        if eval_times is None:
            return out[1] 
        else:
            return out

    def trajectory(self, x, timesteps):
        integration_time = torch.linspace(0., self.T, timesteps)
        return self.forward(x, eval_times=integration_time)

class NeuralODE(nn.Module):
    """
    Returns the flowmap of the neural ODE, i.e. x\mapsto\Phi_T(x), 
    where \Phi_T(x) might be the solution to the neural ODE, or the
    solution composed with a projection. 
    
    ***
    - output dim is an int the dimension of the labels.
    - architecture is a string designating the structure of the dynamics f(x,u)
    - fixed_projector is a boolean indicating whether the output layer is trained or not
    ***
    """
    def __init__(self, device, data_dim, hidden_dim, output_dim=2,
                 augment_dim=0, non_linearity='tanh',
                 tol=1e-3, adjoint=False, architecture='inside', 
                 T=10, time_steps=10, 
                 cross_entropy=True, fixed_projector=False):
        super(NeuralODE, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        if output_dim == 1 and cross_entropy: 
            #output_dim = 1 pour MSE; >=2 pour cross entropy for binary classification.
            raise ValueError('Incompatible output dimension with loss function.')
        self.output_dim = output_dim
        self.tol = tol
        self.T = T
        self.time_steps = time_steps
        self.architecture = architecture
        self.cross_entropy = cross_entropy
        self.fixed_projector = fixed_projector

        dynamics = Dynamics(device, data_dim, hidden_dim, augment_dim, non_linearity, architecture, self.T, self.time_steps)
        self.flow = Semiflow(device, dynamics, tol, adjoint, T,  time_steps)
        self.linear_layer = nn.Linear(self.flow.dynamics.input_dim,
                                         self.output_dim)
        self.non_linearity = nn.Tanh()
        
    def forward(self, x, return_features=False):
        
        features = self.flow(x)

        if self.fixed_projector: #currently fixed_projector = fp
            import pickle
            with open('text.txt', 'rb') as fp:
                projector = pickle.load(fp)
            pred = features.matmul(projector[-2].t()) + projector[-1]
            pred = self.non_linearity(pred)
            self.proj_traj = self.flow.trajectory(x, self.time_steps)

        else:
            self.traj = self.flow.trajectory(x, self.time_steps)
            pred = self.linear_layer(features)
            self.proj_traj = self.linear_layer(self.traj)
            if not self.cross_entropy:
                pred = self.non_linearity(pred)
                self.proj_traj = self.non_linearity(self.proj_traj)
        
        if return_features:
            return features, pred
        return pred, self.proj_traj
