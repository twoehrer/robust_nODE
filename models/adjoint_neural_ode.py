#!/usr/bin/env python3
# -*- coding: utf-8 -*-


##------------#
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from custom_nonlinear import tanh_prime
MAX_NUM_STEPS = 1000

# Useful dicos:
activations = {'tanh_prime': tanh_prime }

#dummy x, needs to be replaced with trajectory of solution of dot(x) = f(u(t),x(t))

time_steps = 10
data_dim = 2
dummy_x = torch.ones([time_steps, data_dim])

class adj_Dynamics(nn.Module):
    """
    Structure of the nonlinear, right hand side $f(u(t), x(t)) of the neural ODE.
    We distinguish the different structures defined in the dictionary "architectures" just above.
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0, 
                non_linearity='tanh_prime', T=10, time_steps=10):
        super(adj_Dynamics, self).__init__()
        self.device = device

        self.data_dim = data_dim
        self.input_dim = data_dim
        self.hidden_dim = hidden_dim

   
        self.non_linearity = activations[non_linearity]
    
        self.T = T
        self.time_steps = time_steps
        
       
        blocks1 = [nn.Linear(self.input_dim, hidden_dim) for _ in range(self.time_steps)]
        self.fc1_time = nn.Sequential(*blocks1)
            
    def forward(self, t, x=dummy_x):
        """
        The output of the class -> t mapsto f(x(t), u(t)) where t is a number.
        """
        dt = self.T/self.time_steps
        k = int(t/dt)

        
        w_t = self.fc1_time[k].weight
        b_t = self.fc1_time[k].bias
        
        #calculation of -Dxf(u(t),x(t))

        out = x.matmul(w_t.t())+b_t
        out = self.non_linearity(out) #this should have dimension d
        out = torch.diag(out) #this should make a diagonal d times d matrix out of it
        out = out.matmul(w_t.t()) #prime_simga matrix times weights
        out = - out

        
        return out




