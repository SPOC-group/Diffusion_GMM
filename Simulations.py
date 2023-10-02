import numpy as np
import random
import torch
from scipy.special import erf
from math import*
from scipy.integrate import quad as itg
from torch.utils.data import DataLoader, Dataset

import argparse
parser=argparse.ArgumentParser(description="Job launcher")

parser.add_argument("-n",type=int)    
parser.add_argument("-m",type=float)  #norm of mu. Set to 1. in the whole manuscipt.
parser.add_argument("-s",type=float)  #sigma

args=parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


nrm=args.m
sigma=args.s

d=50000  #dimension of the data


n_list=[2,4,6,8,16,32,64,128,256,512,1024]

ntot=n_list[args.n]   #number of samples

gen0 = torch.Generator().manual_seed(42)
gen1 = torch.Generator().manual_seed(67)
gen3 = torch.Generator().manual_seed(4122)



xi=torch.randn(ntot,d,generator=gen0)  #x_0^\mu. We fixed the random seed for reproducibility
eta=torch.randn(ntot,d,generator=gen1) #z^\mu. We fixed the random seed for reproducibility

s=torch.sign(torch.randn(ntot,generator=gen3)[:ntot])   #s^\mu 

xi_tot=torch.sum(xi.T*s,1).flatten()/ntot  #xi vector
eta_tot=torch.sum(eta.T*s,1).flatten()/ntot #eta vector
xi_tot=xi_tot.numpy()
eta_tot=eta_tot.numpy()

μ=torch.ones(d)
mu=μ.numpy()

summary={"Mag":[],    "Mag_std":[],"t":[],"MagXi":[],"MagEta":[], "Cosine":[],"Norm":[]}



Nsteps=100   #number of discretization steps for the ODE
dt=1/Nsteps  #time step


#Schedule functions
def alpha(t):
  return t
def beta(t):
  return 1-t

#
def get_x(n,μ,σ,test=False):
    x=s.reshape(n,1)@μ.reshape(1,d)
    x+=eta*σ
    return x


def get_y(n,μ,σ,t,test=False):
    x=get_x(n,μ,σ,test=test)
    y=x*alpha(t)+xi*beta(t)
    return x.to(device),y.to(device)

class generate_data(Dataset):   #data loader object
  def __init__(self,n,μ,sigma=.5,t=.5, test=False):
    self.X,self.Y=get_y(n,μ,sigma,t,test)
    self.μ=μ
    self.sigma=sigma
    self.t=t
    self.samples=n

  def __getitem__(self,idx):
    return self.X[idx].to(device),self.Y[idx].to(device)

  def __len__(self):
    return self.samples


class AE_tied(torch.nn.Module): #DAE
    def __init__(self, d):
        super(AE_tied, self).__init__()

        self.b=torch.nn.Parameter(torch.Tensor([1]))   #skip connection
        self.w=torch.nn.Parameter(torch.randn(d))      #network weight

    def forward(self, x):
        identity=x
        h=torch.sign(x@self.w/np.sqrt(d))
        yhat = h.reshape(x.shape[0],1)@self.w.reshape(1,d)
        yhat+=self.b*identity
        return yhat

def quadloss(ypred, y):      #Loss function. The regularization enters at the level of the optimizer as the weight decay
    return torch.sum((ypred-y)**2)/2

def train(train_loader, t):
    global X
    ae=AE_tied(d).to(device)

    optimizer = torch.optim.Adam([{'params': [ae.w],"weight_decay":1e-1},{'params': [ae.b],"weight_decay":0.}],lr=.04)    
    ######## Training the DAE
    for tt in range(6000):
        for x,y in train_loader:   #Optimization steps
          
          y_pred = ae(y)
          loss = quadloss(y_pred,x)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
    ###### Computing the corresponding velocity
    β=beta(t)
    α=alpha(t)
    
    w_=ae.w.cpu().detach().numpy()
    c=float(ae.b)

    bhat=(np.sign(X@w_).reshape(-1,1))@w_.reshape(1,-1)/β

    lin=(1+α/β)*(X*c)-X/β
    if β==0:
        lin=c*X
    v=(α+β)*bhat+lin
    ####### ODE step
    X+=v*dt

    Mt=((X.T*np.sign(X@mu)).T)@mu/d
    MXi=((X.T*np.sign(X@mu)).T)@xi_tot/d
    MEta=((X.T*np.sign(X@mu)).T)@eta_tot/d/sigma
        
    X_=(X.T*np.sign(X@mu)).T
    Simi=X_@mu/np.sqrt(d)/np.sqrt(np.sum(X_**2, 1))
    summary["Mag"].append(Mt.mean()); summary["Mag_std"].append(Mt.std()); summary["t"].append(t+dt); summary["MagXi"].append(MXi.mean())
    summary["MagEta"].append(MEta.mean()); summary["Cosine"].append(Simi.mean()); summary["Norm"].append(np.sum(X_**2)/X_.shape[0]/d)


    


N=1000
X=np.random.randn(N,d)   #N samples which will be transported by the flow. At time 0, X~\rho_0
                         #At time t=1, X~\hat{\rho}_1

Mt=((X.T*np.sign(X@mu)).T)@mu/d
MXi=((X.T*np.sign(X@mu)).T)@xi_tot/d
MEta=((X.T*np.sign(X@mu)).T)@eta_tot/d

X_=(X.T*np.sign(X@mu)).T
Simi=X_@mu/np.sqrt(d)/np.sqrt(np.sum(X_**2, 1))
summary["Mag"].append(Mt.mean()); summary["Mag_std"].append(Mt.std()); summary["t"].append(0); summary["MagXi"].append(MXi.mean())
summary["MagEta"].append(MEta.mean()); summary["Cosine"].append(Simi.mean()); summary["Norm"].append(np.sum(X_**2)/X_.shape[0]/d)




ts=np.linspace(0.,1,Nsteps)[:-1]

for t in ts:
  X_train=generate_data(ntot,μ,sigma=sigma,t=t)
  train_loader=DataLoader(X_train,batch_size=int(ntot))

  train(train_loader,t)
   


mu_simu=((X.T*np.sign(X@mu)).T).mean(axis=0)   #computing \hat{\mu}, the cluster mean of the estimated density




np.save("data/mu_n{}_norm{}_sig{}.npy".format(ntot,nrm,sigma),mu_simu)

