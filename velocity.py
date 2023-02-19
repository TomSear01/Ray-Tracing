# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device == 'cuda': 
    print(torch.cuda.get_device_name()) 
    

def plotTraining(x,y,v):
    x_plot =x.squeeze(1) 
    y_plot =y.squeeze(1)
    X,Y= np.meshgrid(x_plot,y_plot)
    F_xy = v
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X,Y, F_xy,20,cmap="rainbow")
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('F(x,y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    

#Domain
xUB = 1 #x upper bound
yUB = 1 #y lower bound
nx = 1001
ny = 1001

x = np.linspace(0,xUB,nx)
y = np.linspace(0,yUB,ny)
x = np.expand_dims(x,1)
y = np.expand_dims(y,1)

# Create the mesh 
X,Y=np.meshgrid(x,y)

#Velocity
v = np.zeros_like(X)

##Layered structure
#v[0:333,:] = 0.5
#v[333:666,:] = 1.0
#v[666:-1] = 1.5
#v[ny-1,:] = 1.5

#Gaussian Pulse
#v = 2*np.exp(-1*(((X-0.5)**2)/(2*0.1**2)+((Y-0.5)**2)/(2*0.5**2)))+0.5 

#MULTIPLE Pules
#MULTIPLE Pules
p1 = 2*np.exp(-1*(((X-0.3)**2)/(2*0.1**2)+((Y-0.5)**2)/(2*0.2**2)))+0.5 
p2 = 2*np.exp(-1*(((X-0.7)**2)/(2*0.1**2)+((Y-0.7)**2)/(2*1**2)))+0.5 #3
p3 = 2*np.exp(-1*(((X-0.5)**2)/(2*0.09**2)+((Y-0.1)**2)/(2*0.09**2)))+0.5 #4*...
p4 = 1*np.exp(-1*(((X-0.5)**2)/(2*0.5**2)+((Y-0.5)**2)/(2*0.1**2)))+0.5 #3
#v = p1+p2+p3+p4

#Hyperbolic Tangent
v = 0.2*(np.tanh(100*X-50) + 0*Y)+1.5


plotTraining(x,y,v)


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
        return a
                        
    def loss(self,x,y):              
        loss = self.loss_function(self.forward(x), y)              
        return loss


xy_test = np.hstack((X.flatten()[:,None], Y.flatten()[:,None])) 
v_true = v.flatten('F')[:,None]

#To Tensor
xy_train = torch.from_numpy(xy_test).float().to(device)
v_train = torch.from_numpy(v_true).float().to(device)

#Domain Bounds
lb = xy_test[0]
ub = xy_test[-1]


lr=1e-2
layers = np.array([2,200,200,200,200,1]) #8 hidden layers

total_loss = []
epoch_count = []

model = NN(layers)
model.to(device)

optimiser = torch.optim.Adam(model.parameters(),lr=lr,amsgrad=False)

num_samples = 20000

for i in range(2): # reducing learning rate after each
    print(f"\nLoop: {i+1}, lr:{lr}")
    for epoch in range(20000):
        
        
        #Train eith 20,000 random points each epoch
        inx = np.random.choice(xy_test.shape[0],num_samples, replace=False) 
        
        xy_train_lp = xy_train[inx,:] #Get training data
        v_train_lp = v_train[inx]
        
        optimiser.zero_grad()
        loss = model.loss(xy_train_lp,v_train_lp)
 
       
        
        if epoch%1000==0:
            epoch_count.append(epoch)
            total_loss.append(loss.detach().to('cpu').numpy())
            print(f"Epoch: {epoch} |total loss: {loss}")
       
        loss.backward()
        optimiser.step()
           
    for grp in optimiser.param_groups:
        grp['lr']*=0.5
        lr = grp['lr']
        
torch.save(model,'velocity_tanh.pth')

v_test = model.forward(xy_train).detach()

v_test = v_test.to('cpu')
v_test = torch.reshape(v_test,(1001,1001)).T

v_diff = v_test-v

plotTraining(x,y,v_test)

plotTraining(x,y,v_diff)


















