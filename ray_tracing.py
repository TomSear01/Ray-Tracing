# -*- coding: utf-8 -*-
'''
PINN for seismic ray tracing. 
Velocity field is described by a neural network

'''


import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.autograd as autograd
import torch.nn as nn
import time
from scipy import integrate


torch.set_default_dtype(torch.float)

#Set random number generators
torch.manual_seed(80701)
np.random.seed(80701)

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
    plt.scatter((src[0],rcvr[0]),(src[1],rcvr[1]),marker='o', c='k', linewidths=1.5)
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
    cp = ax.contourf(X,Y, F_xy,20,cmap="RdBu")
    #plt.scatter((src[0],rcvr[0]),(src[1],rcvr[1])),marker='o', c='k', linewidths=1.5)
    plt.plot(rayx,rayy, c='k')
    fig.colorbar(cp, label='Velocity') # Add a colorbar to a plot
    #plt.text(src[0]+1,src[1],'Source')
    #plt.text(rcvr[0]+1,rcvr[1],'Reciever')
    plt.ylim(0,1)
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
        a = a/10      
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

vel_model = torch.load('velocity_tanh.pth') #Load previously trained velocity NN
vel_model.to(device)
vel_model.eval()

xy_stack = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))
xy_stackT = torch.from_numpy(xy_stack).to(device)
xy_stackT.requires_grad = True  

#Domain Bounds
lb = xy_stack[0]
ub = xy_stack[-1]
 
v = vel_model.forward(xy_stackT)

#v_plot = 0.2*(np.tanh(100*X-50) + 0*Y)+1.5
#v_plot = v_plot.T

v_plot = v.to('cpu').detach().numpy()
v_plot = np.reshape(v_plot,(1001,1001))

#Source
src = np.array([0.05,0.05]) #Location of source
rcvr = np.array([0.95,1]) #Location of reciever (now on surface!)

plotTraining(x,y,v_plot,src,rcvr) 

#Ray
Lambda = np.linspace(0,1,101)
LambdaT = torch.from_numpy(Lambda).unsqueeze(1).to(device)


srcT = torch.from_numpy(src).to(device)
rcvrT = torch.from_numpy(rcvr).to(device)


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
        
        
        a = (1-L)*srcT+L*rcvrT+L*(1-L)*a #Enforce output to source and reciever
        
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
        drLy_L = autograd.grad(rLy, L, torch.ones_like(rLy).to(device),retain_graph=True, create_graph=True)[0] #d r(lambda)y coordiante / dLambda
        
        mulx = slow*drLx_L 
        muly = slow*drLy_L
        
        
        d_mulx = autograd.grad(mulx, L, torch.ones_like(mulx).to(device), retain_graph=True, create_graph=True)[0]
        d_muly = autograd.grad(muly, L, torch.ones_like(muly).to(device), retain_graph=True, create_graph=True)[0]
        
        eqnx = d_mulx-dslow_x
        eqny = d_muly-dslow_y
        
        eqn = torch.cat((eqnx,eqny),1) # Put x and y equations back into one tensor

        
        
        loss_f = (eqn**2).mean()  #Mean squared         
        return loss_f



lr=1e-3
layers = np.array([1,20,20,20,20,20,20,20,20,20,20,2]) #8 hidden layers (tried with only 2 hidden layers and didn't work :( )

total_loss = []
epoch_count = []

PINN = pinn(layers)
PINN.to(device)


optimiser = torch.optim.Adam(PINN.parameters(),lr=lr,amsgrad=False)

#Plot PINN before training
ray = PINN.forward(LambdaT)
ray = ray.to('cpu').detach().numpy()

plotResult(x,y,v_plot,src,rcvr,ray)

num_samples = 25

st = time.time()

for i in range(1): # reducing learning rate after each
    print(f"\nLoop: {i+1}, lr:{lr}")
    for epoch in range(3000):

        optimiser.zero_grad()
        
        #Train eith 20,000 random points each epoch
        #inx = np.random.choice(LambdaT.shape[0],num_samples, replace=False)
        LambdaT_in = LambdaT#[inx]
        
        
        loss = PINN.loss_PDE(LambdaT_in)
         
        if epoch%1000==0:
            epoch_count.append(epoch)
            total_loss.append(loss.detach().to('cpu').numpy())
            print(f"Epoch: {epoch} |total loss: {loss}")
       
        loss.backward()
        optimiser.step()

        
    if lr == 1e-3 and i%2==0:# and i > 1:    #1e-5
      for grp in optimiser.param_groups:
        grp['lr']*=0.1
        lr = grp['lr']
    
    
et = time.time()

elapsed_time = et-st
print('Execution time:', elapsed_time, 'Seconds')


ray = PINN.forward(LambdaT)
ray = ray.to('cpu').detach().numpy()

plotResult(x,y,v_plot,src,rcvr,ray)

device = 'cpu'
vel_model.to('cpu')
PINN= PINN.to('cpu')
srcT = torch.from_numpy(src).to('cpu')
rcvrT = torch.from_numpy(rcvr).to('cpu')

#Travel Time
def slow(L):
  L = np.array([L])
  L = torch.from_numpy(L)
  L.requires_grad=True
  ray = PINN.forward(L)
  X_dL = autograd.grad(ray,L,torch.ones_like(ray))[0][0]
  v = vel_model.forward(ray).detach().numpy()
  #v = 0*ray[0] + 0*ray[1] +2
  s = v**-1
  out = s*X_dL.detach().numpy()
  return out

time = integrate.quad(slow,0.,1.)
print(time[0])





