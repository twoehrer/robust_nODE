import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

x_blocks = [nn.Linear(2, 2) for _ in range(10)]
dummy_x = nn.Sequential(*x_blocks)

print(dummy_x[0].weight)

t = torch.tensor([0,1,2])
print(t)

print(torch.flip(t,[0]))

print(t.type())




import torch
device = torch.device('cpu')
from torch.utils.data import DataLoader
import pickle


##--------------#
## Setup:
hidden_dim, data_dim = 2, 2
T, num_steps = 5.0, 10
dt = T/num_steps
turnpike = False
bound = 0.
fp = False
cross_entropy = True



class Semiflow(nn.Module):  #this is what matters
    """
    Given the dynamics f, generate the semiflow by solving x'(t) = f(u(t), x(t)).
    We concentrate on the forward Euler method - the user may change this by using
    other methods from torchdiffeq in the modules odeint and odeint_adjoint.

    ***
    - dynamics denotes the instance of the class Dynamics, defining the dynamics f(x,u)
    ***
    """
    def __init__(self, device, dynamics, adj_dynamics, tol=1e-3, adjoint=False, T=10, time_steps=10):
        super(Semiflow, self).__init__()
        self.adjoint = adjoint #here there is already an implementation of adjoint for backwards propagation
        self.device = device
        self.dynamics = dynamics
        self.tol = tol
        self.T = T
        self.time_steps = time_steps
        
        self.adj_dynamics = adj_dynamics


    def forward(self, x, eval_times=None): #i probably need to add a p argument here
    
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

        if self.adjoint:  
            out = odeint_adjoint(self.dynamics, x_aug, integration_time, method='euler', options={'step_size': dt})
        else:
            out = odeint(self.dynamics, x_aug, integration_time, method='euler', options={'step_size': dt})
            adj_out = odeint(self.adj_dynamics, torch.eye(x.shape[0]), torch.flip(integration_time,[0]), method='euler', options={'step_size': dt}) #this is new for the adjoint
            print(out.type())
        if eval_times is None:
            return out[1] 
        else:
            return out, adj_out

    def trajectory(self, x, timesteps):
        integration_time = torch.linspace(0., self.T, timesteps)
        return self.forward(x, eval_times=integration_time)


flow = Semiflow(device, dynamics, adj_dynamics, tol, adjoint, T,  time_steps)