# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:07:46 2023

@author: C301654
"""


import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense



dataset=pd.read_csv('simple_linear.csv')

x_train=dataset[['X1','X2']]

y_train=dataset['out']


'''*******************Keras******************************'''

regressor=Sequential()

regressor.add(Dense(input_dim=2,output_dim=10,init='uniform',activation='relu'))
regressor.add(Dense(output_dim=20,activation='relu'))
regressor.add(Dense(output_dim=10,activation='relu'))
regressor.add(Dense(output_dim=1))

regressor.compile(optimizer='adam',loss='mse')

regressor.fit(x_train,y_train,nb_epoch=1000)


y_pred=regressor.predict(x_train)


'''*******************pytorch******************************'''

import torch
import torch.nn as nn

X=dataset[['X1','X2']]
y=dataset['out']

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
#model=model.cuda()


#loss and optim
loss1=nn.MSELoss()
optim1=torch.optim.SGD(model.parameters(),lr=0.0001)

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
    
    if epoch%10==0:
        print(f'epoch {epoch+1},loss={l}')



with torch.no_grad():
    y_predicted=model(x_train)
    print(y_predicted)