# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 20:10:38 2023

@author: S Mahesh Reddy
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:38:56 2023

@author: S Mahesh Reddy
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

dataset=pd.read_csv("simple_linear.csv")

dataset.head()

dataset.isna().sum()

X=dataset[['X1','X2']]

y=dataset['out']

X.isna().sum()

x_numpy=X.to_numpy().reshape(-1,2)
y_numpy=y.to_numpy().reshape(-1,1)

X_tensor=torch.from_numpy(x_numpy.astype(np.float32))
y_tensor=torch.from_numpy(y_numpy.astype(np.float32))

X_tensor[0,:]

class ANN_first(nn.Module):
    def __init__(self,input_dim):
        super(ANN_first,self).__init__()
        self.l1=nn.Linear(input_dim,1)
        #self.r1=nn.ReLU()
        #self.l2=nn.Linear(10,20)
        #self.r2=nn.ReLU()
        #self.l3=nn.Linear(20,10)
        #self.r3=nn.ReLU()
        #self.l4=nn.Linear(10,1)

    def forward(self,x):
        out=self.l1(x)
        #out=self.r1(out)
        #out=self.l2(out)
        #out=self.r2(out)
        #out=self.l3(out)
        #out=self.r3(out)
        #out=self.l4(out)
        return out

input_dim=x_numpy.shape[1]


model=ANN_first(input_dim)

for i in model.parameters():
    print(i.is_cuda)
model=model.cuda()


#loss and optim
loss1=nn.MSELoss()
optim1=torch.optim.SGD(model.parameters(),lr=1e-5)

#training loop


if(torch.cuda.is_available()):
    device='cuda'
else:
    device='cpu'
    


batch_size=10
n_epochs=3000




from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

dataset_train = TensorDataset(X_tensor, y_tensor)


# how assign
train_loader = DataLoader(dataset=dataset_train, batch_size=50)
training_losses = []
for epoch in range(n_epochs):
    batch_losses = []
    for nbatch, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        y_pred=model(x_batch)
        l=loss1(y_pred,y_batch)
        
        optim1.zero_grad()
        l.backward()
        optim1.step()
        batch_losses.append(l.item())
    training_loss = np.mean(batch_losses)
    training_losses.append(training_loss)
    
    if((epoch+1)%10==0):
        print(f'epoch {epoch+1},Training loss: {training_loss:.4f}')



with torch.no_grad():
    y_predicted=model(X_tensor.to(device))
    print(y_predicted)
        
        
print("all good")

'''*****************************Pytorch Kaggle tutorial*****************************'''



dataset_train = TensorDataset(X_tensor, y_tensor)
# how assign
train_loader = DataLoader(dataset=dataset_train, batch_size=50)
'''
def make_train_step(model, loss_fn, optimizer):
    # builds & returns the function that will be called inside the loop
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step'''


device = 'cpu'

# hyperparameters
lr = 1e-5
n_epochs = 10000

from sklearn.metrics import r2_score

# loss function & optimizer
model = (nn.Linear(2, 1)).to(device)
loss_fn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


# training step
#train_step = make_train_step(model, loss_fn, optimizer)
training_losses = []
test_losses = []
accuracies = []
for epoch in range(n_epochs):
    batch_losses = []
    for nbatch, (x_batch, y_batch) in enumerate(train_loader):
        #print(nbatch)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        #loss = train_step(x_batch, y_batch)
        yhat = model(x_batch)
        loss = loss_fn(y_batch, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_losses.append(loss.item())
    training_loss = np.mean(batch_losses)
    training_losses.append(training_loss)
    # 
    
    if epoch % 50 == 0:
        print(f'epoch {epoch+1} | Training loss: {training_loss:.4f} ')



with torch.no_grad():
    y_predicted=model(X_tensor.to(device))
    print(y_predicted)





        
        


'''********************************Old Code*************************************'''

dataset=pd.read_csv("simple_linear.csv")

dataset.head()

dataset.isna().sum()

X=dataset[['X1','X2']]

y=dataset['out']

X.isna().sum()

x_numpy=X.to_numpy()
y_numpy=y.to_numpy()

X_tensor=torch.from_numpy(x_numpy.astype(np.float32))
y_tensor=torch.from_numpy(y_numpy.astype(np.float32))

X_tensor[0,:]

class ANN_first(nn.Module):
    def __init__(self,input_dim):
        super(ANN_first,self).__init__()
        self.l1=nn.Linear(input_dim,10)
        self.r1=nn.ReLU()
        self.l2=nn.Linear(10,20)
        self.r2=nn.ReLU()
        self.l3=nn.Linear(20,10)
        self.r3=nn.ReLU()
        self.l4=nn.Linear(10,1)

    def forward(self,x):
        out=self.l1(x)
        out=self.r1(out)
        out=self.l2(out)
        out=self.r2(out)
        out=self.l3(out)
        out=self.r3(out)
        out=self.l4(out)
        return out

input_dim=x_numpy.shape[1]


model=ANN_first(input_dim)

for i in model.parameters():
    print(i.is_cuda)
model=model.cuda()


#loss and optim
loss1=nn.MSELoss()
optim1=torch.optim.SGD(model.parameters(),lr=0.001)

#training loop


if(torch.cuda.is_available()):
    device='cuda'
else:
    device='cpu'
    
x_train=X_tensor.to(device)
y_train=y_tensor.to(device)

batch_size=10

n_epochs=10000

for epoch in range(n_epochs):
    
    for iter in range(0,220,batch_size):
        x_train=X_tensor[iter:iter+batch_size,:].to(device)
        y_train=y_tensor[iter:iter+batch_size].to(device)
        
        y_pred=model(x_train)
        l=loss1(y_pred,y_train)
        
        optim1.zero_grad()
        l.backward()
        optim1.step()
        
        if((epoch+1)%10==0):
            print(f'epoch {epoch+1},loss={l}')



with torch.no_grad():
    y_predicted=model(x_train)
    print(y_predicted)
        
        
print("all good")
        
        
        
        