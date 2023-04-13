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
        self.l1=nn.Linear(input_dim,1)
        

    def forward(self,x):
        out=self.l1(x)
        
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



n_epochs=1000

for epoch in range(n_epochs):
    
    y_pred=model(x_train)
    l=loss1(y_pred,y_train)
    
    optim1.zero_grad()
    l.backward()
    optim1.step()
    
    
    print(f'epoch {epoch+1},loss={l}')



with torch.no_grad():
    y_predicted=model(x_train)
    print(y_predicted)
        
        
        
        
        
        
        