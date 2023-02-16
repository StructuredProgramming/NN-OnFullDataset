from __future__ import print_function
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import os
import ast
import numpy as np
import pywt
import random
import matplotlib.pyplot as plt
from PIL import Image as PImage
device="cpu"
def weights_init(m):
    if isinstance(m, nn.Linear):
      torch.nn.init.kaiming_uniform_(m.weight)
    
#Plan to run twelve versions and examine the loss: 22->720, 22->360, 22->180, 22->90, 36->720, 36->360, 36->180, 36->90, 60->720, 60->360, 60->180, 60->90, this will determine which weights will be used for the Neural Network
class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()

        # Encoder
        self.encode=nn.Sequential(
        nn.Linear(22,120),
        nn.ReLU(),
        nn.Linear(120,120),
        nn.ReLU(),
        nn.Linear(120,120),
        nn.ReLU(),
        nn.Linear(120,120),
        nn.ReLU(),
        nn.Linear(120,120),
        nn.ReLU(),
        nn.Linear(120,120),
        nn.ReLU(),
        nn.Linear(120,120),
        nn.ReLU(),
        nn.Linear(120,120)
        )
        self.fc_mu = nn.Linear(120, z_dim)
        self.fc_logvar = nn.Linear(120, z_dim)
        # Decoder
        self.decode=nn.Sequential(
        nn.Linear(z_dim, 120),
        nn.ReLU(),
        nn.Linear(120, 120),
        nn.ReLU(),
        nn.Linear(120,120),
        nn.ReLU(),
        nn.Linear(120,120),
        nn.ReLU(),
        nn.Linear(120,120),
        nn.ReLU(),
        nn.Linear(120,120),
        nn.ReLU(),
        nn.Linear(120,120),
        nn.ReLU(),
        nn.Linear(120,120),
        nn.ReLU(),
        nn.Linear(120,22)
        )
        nn.init.kaiming_uniform_(self.fc_mu.weight)
        nn.init.kaiming_uniform_(self.fc_logvar.weight)
        self.encode.apply(weights_init)
        self.decode.apply(weights_init)

        
    def encoder(self, x):
        a=self.encode(x)
        return self.fc_mu(a),self.fc_logvar(a)     

    def sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample

    def decoder(self, z):
        return self.decode(z)
       
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sampling(mu, logvar)
        return self.decoder(z), mu, logvar
#Will test 9 different values: 3000, 1536, 1000, 768, 500, 384, 192, 120, 96
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layer,latentdimensions):
        super(NeuralNetwork,self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(latentdimensions,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,8)
            )            
    def forward(self,x):
        output=self.linear_relu_stack(x)
        return output
        
def coords(m):
    for i in range(2, len(m) - 3):
        if m[i] == ',':
            return m[2:i],m[(i+2):(len(m)-2)]
trainloss=0
testloss=0
vae = VAE(z_dim=5).double().to(device)
myfinaltrainloss=[]
myfinaltestloss=[]
vae.load_state_dict(torch.load("Weights120-22.txt", map_location=torch.device('cpu')))
loss_function2=nn.MSELoss()
model=NeuralNetwork(hidden_layer=256, latentdimensions=5).double().to(device)   
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
batch_size=128
with open('ISEFDataset.txt', 'r') as f: 
    lines = f.readlines()
for epoch in range (200):
    print("Epoch number " + str(epoch))
    count=0
    trainloss=0
    testloss=0
    itertrain=0
    itertest=0
    runningnum=0
    random.shuffle(lines)
    for line in lines:
        count+=1
        x, y = line.split('=')[0], line.split('=')[1]
        a=line.split('=')
        if len(a)==6:
            movingjoint1=a[2]
            movingjoint2=a[3]
            movingjoint3=a[4]
            movingjoint4=a[5]
            x1,y1=coords(movingjoint1)
            x2,y2=coords(movingjoint2)
            x3,y3=coords(movingjoint3)
            x4,y4=coords(movingjoint4)
        else:
            movingjoint1=a[3]
            movingjoint2=a[5]
            movingjoint3=a[6]
            movingjoint4=(0,0)
            x1,y1=coords(movingjoint1)
            x2,y2=coords(movingjoint2)
            x3,y3=coords(movingjoint3)
            x4,y4=coords(movingjoint4)
        x, y = x.split(' '), y.split(' ')
        x = [i for i in x if i]
        y = [i for i in y if i]
        x[0] = x[0][1:]
        y[0] = y[0][1:]
        x[-1] = x[-1][:-1]
        y[-1] = y[-1][:-1]

        x = [float(i) for i in x if i]
        y = [float(i) for i in y if i]
        if(math.isnan(x[30])):
                continue
        #Fourier Descriptor extraction will depend on which VAE ends up being the best (the number of inputs needed could vary as a result)
        S=np.zeros(360, dtype='complex_')
        i=0
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=359
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=1
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=358
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=2
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=357
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=3
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=356
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=4
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=355
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=5
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        #Input representation will differ based on VAE used
        input_list=[float(np.real(S[355])),float(np.real(S[356])),float(np.real(S[357])),float(np.real(S[358])), float(np.real(S[359])), float(np.real(S[0])), float(np.real(S[1])),float(np.real(S[2])),float(np.real(S[3])),float(np.real(S[4])),float(np.real(S[5])), float(np.imag(S[355])),float(np.imag(S[356])),float(np.imag(S[357])), float(np.imag(S[358])), float(np.imag(S[359])), float(np.imag(S[0])), float(np.imag(S[1])), float(np.imag(S[2])), float(np.imag(S[3])),float(np.imag(S[4])),float(np.imag(S[5]))]
        input_tensor=torch.tensor(input_list).double().to(device)
        latent_vector=vae.encoder(input_tensor)
        prediction=model(latent_vector[0])
        #print(latent_vector[0])
        #print(prediction)
        output_list=[float(x1),float(x2),float(x3),float(x4),float(y1),float(y2),float(y3),float(y4)]
        output_tensor=torch.tensor(output_list).double().to(device)
        loss_function=nn.MSELoss()
        loss=loss_function(prediction,output_tensor)
        #print(prediction)
        #print(output_tensor)
        runningnum+=1
        if(runningnum<2554):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            myloss=loss.item()
            if(count%100==0):
                count=0
                print(myloss)
            itertrain+=1
            trainloss+=loss
            #print(prediction)
            #print(output_tensor)
        elif(runningnum>=2554):
            count=0
            testloss+=loss
            itertest+=1
    myfinaltrainloss.append(trainloss/itertrain)
    myfinaltestloss.append(testloss/itertrain)
print(myfinaltrainloss)
print(myfinaltestloss)
