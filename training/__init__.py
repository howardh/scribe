import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm

#class MaskFunction(torch.autograd.Function):
#    @staticmethod
#    def forward(self, x, mask):
#        self.save_for_backward(mask)
#        return x*mask.float()
#
#    @staticmethod
#    def backward(self, grad):
#        mask, = self.saved_variables
#        return grad*mask.float(), None

class MaskFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, x, mask):
        self.save_for_backward(mask)
        return x[mask].view(-1)

    @staticmethod
    def backward(self, grad):
        mask, = self.saved_variables
        output = torch.zeros(mask.size()).view(-1)
        if grad.is_cuda:
            output = output.cuda()
        gi = 0
        for i,m in enumerate(mask.view(-1)):
            if m.all():
                output[i] = grad.data.view(-1)[gi]
                gi += 1
        output = Variable(output.view(mask.size()))
        if (grad.size()[0] != torch.sum(mask)).any():
            print("mask ", mask)
            print("grads ", grad)
            print("output ", output)
        return output, None

def print_avg_grad(rnn):
    grads = []
    vals = []
    for p in rnn.parameters():
        grads+= list(np.abs(p.grad.view(-1).data.cpu().numpy()))
        vals += list(np.abs(p.view(-1).data.cpu().numpy()))
    tqdm.write('Average grad: %f, Average vals: %f' % (np.mean(grads), np.mean(vals)))
    return np.mean(grads)


def prob(x, y):
    """
    Return the probability of the next point being x given that parameters y
    were outputted by the neural net.
    See equation (23)
    """
    e,pi,mu,sigma,rho = y
    num_components = mu.size()[2]
    p = 0
    for i in range(num_components):
        p += pi[:,:,i]*normal(x[:,:,1:], mu[:,:,i,:],sigma[:,:,i,:],rho[:,:,i])
    p *= (x[0,:,0]>0.5).float()*e+(x[0,:,0]<0.5).float()*(1-e)
    #temp1 = x[0,:,0]>0.5
    #temp2 = x[0,:,0]<0.5
    #if (temp1==temp2).any():
    #    print("Not working. Fix me.")
    #    print(temp1==temp2)
    #    print(x[0,:,0])
    #p *= torch.where(x[0,:,0]==1,e,1-e)
    #if (p<=0.00000001).any():
    #    if p.is_cuda:
    #        p=Variable(torch.ones(p.size()).cuda(), requires_grad=False)
    #    else:
    #        p=Variable(torch.ones(p.size()), requires_grad=False)
    return p

def normal(x, mu, sigma, rho):
    z  = torch.pow((x[:,:,0]-mu[:,:,0])/sigma[:,:,0],2)
    z += torch.pow((x[:,:,1]-mu[:,:,1])/sigma[:,:,1],2)
    #z = torch.pow((x-mu)/sigma,2)
    z -= 2*rho*(x[:,:,0]-mu[:,:,0])*(x[:,:,1]-mu[:,:,1])/(sigma[:,:,0]*sigma[:,:,1])
    output = 1/(2*np.pi*sigma[:,:,0]*sigma[:,:,1]*torch.sqrt(1-rho*rho))*torch.exp(-z/(2*(1-rho*rho)))
    return output

def create_optimizer(rnn):
    # Paper says they're using RMSProp, but the equations (38)-(41) look like Adam with momentum.
    # See parameters in equation (42)-(45)
    # Reference https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
    # Reference http://pytorch.org/docs/master/optim.html
    # (42) is Wikipedia's beta1 and beta2
    # (43) is momentum
    # (44) is learning rate
    # (45) is epsilon (added to denom for numerical stability)
    # Skipped out on Momentum, since it's not implemented by pytorch
    optimizer = torch.optim.Adam(params=rnn.parameters(),lr=0.0001,betas=(0.95,0.95),eps=0.0001)
    #optimizer = torch.optim.RMSprop(params=rnn.parameters(),lr=0.0001,alpha=0.95,eps=0.0001)
    #optimizer = torch.optim.SGD(params=rnn.parameters(),lr=0.0001)
    return optimizer

