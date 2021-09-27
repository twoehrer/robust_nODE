import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

x_blocks = [nn.Linear(2, 2) for _ in range(10)]
dummy_x = nn.Sequential(*x_blocks)

print(dummy_x[0].weight)

t = torch.tensor([0,1,2])
print(t)

print(torch.flip(t,[0]))