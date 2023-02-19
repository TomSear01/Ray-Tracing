# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:49:49 2023

@author: thoma
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.autograd as autograd
import torch.nn as nn
import time

torch.set_default_dtype(torch.float)

#Set random number generators
torch.manual_seed(80701)
np.random.seed(100)

#Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device == 'cuda': 
    print(torch.cuda.get_device_name()) 

#device = 'cpu'
#--------------------------- Plotting functions--------

def plotTraining(x,y,v,src,rcvr):
    x_plot =x.squeeze(1) 
    y_plot =y.squeeze(1)
    X,Y= np.meshgrid(x_plot,y_plot)
    F_xy = v
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X,Y, F_xy,20,cmap="rainbow")
    plt.scatter((src[:,0],rcvr[:,0]),(src[:,1],rcvr[:,1]),marker='o', c='k', linewidths=1.5)
    fig.colorbar(cp, label='Velocity') # Add a colorbar to a plot
    #plt.text(src[0]+1,src[1],'Source')
    #plt.text(rcvr[0]+1,rcvr[1],'Reciever')
    ax.set_title('F(x,y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

def plotResult(x,y,v,src,rcvr,ray):
    x_plot =x.squeeze(1) 
    y_plot =y.squeeze(1)
    X,Y= np.meshgrid(x_plot,y_plot)
    F_xy = v
    rayx = ray[:,0]
    rayy = ray[:,1]
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X,Y, F_xy,20,cmap="rainbow")
    plt.scatter((src[:,0],rcvr[:,0]),(src[:,1],rcvr[:,1]),marker='o', c='k', linewidths=1.5)
    plt.scatter(rayx,rayy, c='k', marker = '.', linewidths = 0.5)
    fig.colorbar(cp, label='Velocity') # Add a colorbar to a plot
    #plt.text(src[0]+1,src[1],'Source')
    #plt.text(rcvr[0]+1,rcvr[1],'Reciever')
    ax.set_title('F(x,y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    
class NN(nn.Module):
    def __init__(self,layers):
        super().__init__() #call __init__ from parent class         
        'activation function'
        self.activation = nn.Softmax()
        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')    
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])        
        self.iter = 0    
        self.layers = layers
        
        'Xavier Normal Initialization'
        for i in range(len(layers)-1):            
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=2)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
            
    'foward pass'
    def forward(self,x):
        
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                        

        
        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)
                      
        #preprocessing input 
        x = (x - l_b)/(u_b - l_b) #feature scaling
        
        #convert to float
        a = x.float()
        
        for i in range(len(self.layers)-2):           
            z = self.linears[i](a)                    
            a = self.activation(z)       
        a = self.linears[-1](a)
        #a = torch.clamp(a,min=0, max = 1.5)        
        return a

                        
    def loss(self,x,y):              
        loss = self.loss_function(self.forward(x), y)              
        return loss


   
#-------------------------------------------------------------------------------
#Domain
xUB = 1 #x upper bound
yUB = 1 #y lower bound

x = np.linspace(0,xUB,1001)
y = np.linspace(0,yUB,1001)

x = np.expand_dims(x,1)
y = np.expand_dims(y,1)

# Create the mesh 
X,Y=np.meshgrid(x,y)

#Velocity Field
layers_vel = np.array([2,200,200,200,200,1]) #8 hidden layers

vel_model = NN(layers_vel) #Initialise Velocity NN

vel_model = torch.load('velocity_Gaussian_smol.pth',map_location=torch.device('cpu')) #Load previously trained velocity NN
vel_model.to(device)
vel_model.eval()

xy_stack = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))


#Domain Bounds
lb = xy_stack[0]
ub = xy_stack[-1]


xy_stackT = torch.from_numpy(xy_stack).to(device) 
v = vel_model.forward(xy_stackT)


v_plot = v.detach().detach().cpu().numpy()
v_plot = np.reshape(v_plot,(1001,1001))


#Imput in to NN: Lambda, src_x, src_y, rcvr_x, rvcr_y




class pinn(nn.Module):
    def __init__(self,layers):
        super().__init__() #call __init__ from parent class         
        'activation function'
        self.activation = nn.Softplus()
        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')    
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])        
        self.iter = 0    
        
        'Xavier Normal Initialization'
        for i in range(len(layers)-1):            
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=0.5)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
            
    'foward pass'
    def forward(self,L):
        
        if torch.is_tensor(L) != True:         
            L = torch.from_numpy(L)                        
        
        #convert to float
        a = L.float()
        
        for i in range(len(layers)-2):           
            z = self.linears[i](a)                    
            a = self.activation(z)       
        a = self.linears[-1](a)
        
        
        a = (1-L[:,0].unsqueeze(1))*L[:,1:3]+L[:,0].unsqueeze(1)*L[:,3:5]+L[:,0].unsqueeze(1)*(1-L[:,0].unsqueeze(1))*a #Enforce output to source and reciever
        
        return a
                        
    def loss_PDE(self, Lambda):                    
        L = Lambda.clone()                  
        L.requires_grad = True    
        rL = self.forward(L) #NN outputs coordinate of ray at point L
              
        vel = vel_model.forward(rL)
        slow = (vel**-1)

        dslow_xy = autograd.grad(slow, rL, torch.ones_like(slow).to(device),retain_graph=True, create_graph=True)[0]
        
        dslow_x = dslow_xy[:,0].unsqueeze(1)
        dslow_y = dslow_xy[:,1].unsqueeze(1)

        rLx = rL[:,0].unsqueeze(1)
        rLy = rL[:,1].unsqueeze(1)

        drLx_L = autograd.grad(rLx, L, torch.ones_like(rLx).to(device),retain_graph=True, create_graph=True)[0] #d r(lambda)x coordiante / dLambda
        drLx_L = drLx_L[:,0].unsqueeze(1)
        
        drLy_L = autograd.grad(rLy, L, torch.ones_like(rLy).to(device),retain_graph=True, create_graph=True)[0] #d r(lambda)y coordiante / dLambda
        drLy_L = drLy_L[:,0].unsqueeze(1)
        
        mulx = slow*drLx_L 
        muly = slow*drLy_L
        
        
        d_mulx = autograd.grad(mulx, L, torch.ones_like(mulx).to(device), retain_graph=True, create_graph=True)[0]
        d_mulx = d_mulx[:,0].unsqueeze(1)
        
        d_muly = autograd.grad(muly, L, torch.ones_like(muly).to(device), retain_graph=True, create_graph=True)[0]
        d_muly = d_muly[:,0].unsqueeze(1)
        
        eqnx = d_mulx-dslow_x
        eqny = d_muly-dslow_y
        
        eqn = torch.cat((eqnx,eqny),1) # Put x and y equations back into one tensor
       
        
        loss_f = (eqn**2).mean()  #Mean squared         
        return loss_f



lr=1e-3
layers = np.array([5,100,100,100,100,100,100,2]) #8 hidden layers #20 Before

train_loss = []
testing_loss = []
epoch_count = []

PINN = pinn(layers)
PINN.to(device)



optimiser = torch.optim.Adam(PINN.parameters(),lr=lr,amsgrad=False)


test = np.zeros((101,5))

test[:,0] = np.linspace(0,1,101)
test[:,1:3] = np.array([0.05,0.05])
test[:,3:5] = np.array([0.95,1])

test = torch.from_numpy(test)
test = test.to(device)

Lambda = np.linspace(0,1,101)

st = time.time()


for i in range(200): # reducing learning rate after each
    print(f"\nLoop: {i+1}, lr:{lr}")
    #---SOURCES,RECIEVERS and RAYS
    n_rays = 100 
    src = np.zeros((n_rays,2))
    rcvr = np.ones((n_rays,2))

    src = np.round(np.random.rand(n_rays,2),2) #Location of source
    rcvr[:,0] = np.round(np.random.rand(n_rays,1),2)[:,0] #Location of reciever (now on surface!)

    srcT = torch.from_numpy(src).to(device)
    rcvrT = torch.from_numpy(rcvr).to(device)
    
    num_samples = 15
    LambdaSR = np.zeros((n_rays*num_samples,5))
    for j in range(n_rays):
      inx = np.random.choice(101,num_samples, replace=False)
      inx = np.sort(inx)
    
      LambdaSR[j*num_samples:num_samples*(j+1),0] = Lambda[inx]
      LambdaSR[j*num_samples:num_samples*(j+1),1:3] = src[j,:]
      LambdaSR[j*num_samples:num_samples*(j+1),3:5] = rcvr[j,:]
    

    LambdaSR_T = torch.from_numpy(LambdaSR)
    LambdaSR_T = LambdaSR_T.to(device)

    
    for epoch in range(1000):

        optimiser.zero_grad()       
        loss = PINN.loss_PDE(LambdaSR_T)
         
        if epoch%10==0:
            #epoch_count.append(epoch)
            train_loss.append(loss.detach().to('cpu').numpy())           
            test_loss = PINN.loss_PDE(test)
            testing_loss.append(test_loss.detach().to('cpu').numpy())
        if epoch%100==0:
          print(f"Epoch: {epoch} |total loss: {loss} | Test Loss: {test_loss}")
            
            
       
        loss.backward()
        optimiser.step()

        
    if lr > 1e-5 and i%25==0:# and i > 1:    #1e-5
      for grp in optimiser.param_groups:
        grp['lr']*=0.1
        lr = grp['lr']
    
    
et = time.time()

elapsed_time = et-st
print('Execution time:', elapsed_time, 'Seconds')


#torch.save(PINN,'mul_gauss_ray_new.pth')

#Final Test
n_rays = 1 #30

src = np.zeros((n_rays,2))
rcvr = np.ones((n_rays,2))

 
src = np.round(np.random.rand(n_rays,2),2) #Location of source
rcvr[:,0] = np.round(np.random.rand(n_rays,1),2)[:,0] #Location of reciever (now on surface!)

srcT = torch.from_numpy(src).to(device)
rcvrT = torch.from_numpy(rcvr).to(device)

#Ray
#Imput in to NN: Lambda, src_x, src_y, rcvr_x, rvcr_y

LambdaSR = np.zeros((n_rays*101,5))
for i in range(n_rays):
    LambdaSR[i*101:101*(i+1),0] = np.linspace(0,1,101)
    LambdaSR[i*101:101*(i+1),1:3] = src[i,:]
    LambdaSR[i*101:101*(i+1),3:5] = rcvr[i,:]
    
    
LambdaSR_T = torch.from_numpy(LambdaSR)
LambdaSR_T = LambdaSR_T.to(device)


ray = PINN.forward(LambdaSR_T)
ray = ray.to('cpu').detach().numpy()

plotResult(x,y,v_plot,src,rcvr,ray)


train_loss = np.asarray(train_loss)
testing_loss = np.asarray(testing_loss)

fig,ax=plt.subplots(1,1)
plt.plot(train_loss, label='Training Loss')
plt.plot(testing_loss, label='Test Loss')
plt.yscale('log')
plt.xlabel(' Training Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()




