from torch import add
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class identityBlock(nn.Module):
    def __init__(self, in_f, *args, **kwargs):
        super().__init__()
        self.id_block = nn.Sequential(
        nn.Linear(in_f,in_f),
        nn.BatchNorm1d(in_f),
        nn.ReLU(),
        nn.Linear(in_f,in_f),
        nn.BatchNorm1d(in_f),
        nn.ReLU(),
        nn.Linear(in_f,in_f),
        nn.BatchNorm1d(in_f)
        )

    def forward(self,x):
       x_orig = x.clone() 
       x = self.id_block(x)
       x = add(x, x_orig)
       x = F.relu(x)
       return x 

class denseBlock(nn.Module):
    def __init__(self, in_f, out_f, *args, **kwargs):
        super().__init__()
        self.dense_block = nn.Sequential(
        nn.Linear(in_f,out_f),
        nn.BatchNorm1d(out_f),
        nn.ReLU(),
        nn.Linear(out_f,out_f),
        nn.BatchNorm1d(out_f),
        nn.ReLU(),
        nn.Linear(out_f,out_f),
        nn.BatchNorm1d(out_f),
        )
        self.projector = nn.Sequential(
            nn.Linear(in_f, out_f),
            nn.BatchNorm1d(out_f)
            )
    def forward(self,x):
       x_orig = x.clone() 
       x_orig = self.projector(x_orig)
       x = self.dense_block(x)
       x = add(x, x_orig)
       x = F.relu(x)
       return x
    
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.dense_block = denseBlock(in_c,out_c)
        self.id_block_1 = identityBlock(out_c)
        self.id_block_2 = identityBlock(out_c)
        
    def forward(self, x):
        x = self.dense_block(x)
        x = self.id_block_1(x)
        x = self.id_block_2(x)
        return x    

class ResidualNetwork(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()

        self.nn = nn.Sequential(
            *[ResBlock(in_f, out_f) for in_f,out_f in zip(in_c[:-1],out_c[:-1])],
            nn.Linear(in_c[-1], out_c[-1]))
            
    def forward(self,x):
        x = self.nn(x)
        return x

model = ResidualNetwork([10,32,64,32,10], [32,64,32,10,1])
print(model)