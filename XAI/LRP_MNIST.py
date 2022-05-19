import utils
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Architecture import CNNMNIST as AMNIST
from DataLoader import dataloaderMNIST as DMNIST
from Test import Validazione as V
import numpy as np
#creiamo una lista in cui memorizzare gli scores di rilevanza ad ogni livello
X,T = utils.loaddata()
utils.digit(X.reshape(1,12,28,28).transpose(0,2,1,3).reshape(28,12*28),9,0.75)

model = AMNIST.CNN().cuda()\
    if torch.cuda.is_available() else AMNIST.CNN()

model.load_state_dict(torch.load('../checkpointCNNMNIST', map_location=torch.device('cpu')))


for i in range (0, 12):
    out=model(DMNIST.test_data.data[i].unsqueeze(0).float())
    img=DMNIST.test_data.data[i]
    plt.imshow(img.squeeze().numpy().astype(np.uint8))
    print(out)

W,B = utils.loadparams()
L = len(W)

R = [None]*L + [out[L]*(T[:,None]==np.arange(10))]

def rho(w,l):
    return w + [None,0.1,0.0,0.0][l] * np.maximum(0,w)

def incr(z,l):
    return z + [None,0.0,0.1,0.0][l] * (z**2).mean()**.5+1e-9


for l in range(1, L)[::-1]:
    w = rho(W[l], l)
    b = rho(B[l], l)

    z = incr(out[l].dot(w) + b, l)  # step 1
    s = R[l + 1] / z  # step 2
c = s.dot(w.T)  # step 3
    R[l] = out[l] * c  # step 4
w  = W[0]
wp = np.maximum(0,w)
wm = np.minimum(0,w)
lb = A[0]*0-1
hb = A[0]*0+1

z = A[0].dot(w)-lb.dot(wp)-hb.dot(wm)+1e-9        # step 1
s = R[1]/z                                        # step 2
c,cp,cm  = s.dot(w.T),s.dot(wp.T),s.dot(wm.T)     # step 3
R[0] = A[0]*c-lb*cp-hb*cm                         # step 4
utils.digit(X.reshape(1,12,28,28).transpose(0,2,1,3).reshape(28,12*28),9,0.75)
utils.heatmap(R[0].reshape(1,12,28,28).transpose(0,2,1,3).reshape(28,12*28),9,0.75)
