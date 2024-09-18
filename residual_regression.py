import torch 
from torch import add
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np 
import matplotlib.pyplot as plt 

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

def get_model(lr):
  model =  ResidualNetwork([12,64,12], [64,12,1])
  #return model, optim.SGD(model.parameters(), lr=lr, momentum =0.9)
  return model, optim.Adam(model.parameters(), lr=lr)

def loss_batch(model, loss_func, xb, yb, opt=None):
  loss = loss_func(model(xb).squeeze(), yb)
  if opt is not None:
      loss.backward()
      opt.step()
      opt.zero_grad()
  return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
  los = []
  model.train()
  #dev = torch.device(
  #  "cpu")
  dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
  for epoch in range(epochs):
    model.train()
    for xb,yb in train_dl:
      xb = xb.to(dev)
      yb = yb.to(dev)
      loss_batch(model, loss_func, xb, yb, opt)

    model.eval()
    with torch.no_grad():
      losses, nums = zip(
          *[loss_batch(model, loss_func, xb.to(dev), yb.to(dev)) for xb, yb in valid_dl]
      )
    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    los.append(val_loss)
    print(epoch, val_loss)
  return los

def get_data(train_ds, valid_ds, bs):
  return (
    DataLoader(train_ds, batch_size=bs, shuffle=True),
    DataLoader(valid_ds, batch_size=bs * 2),
  )
if __name__ == "__main__":
    torch.manual_seed(123)

    lr = 1e-3
    loss_func = F.mse_loss
    dev = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {dev} device")
    #dev = torch.device(
    #    "cpu")
    bs = 32
    epochs = 100
    # Create an instance of the network
    #Need X_Train/Test and Y_train/test 
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    model, opt = get_model(lr = lr )
    model.to(dev)
    los = fit(epochs, model, loss_func, opt, train_dl, valid_dl)
    print(np.shape(los))
    print(np.shape(np.arange(0,epochs,1)))
    import matplotlib.pyplot as plt
    plt.plot(np.arange(0,epochs,1), los)
    plt.show()

