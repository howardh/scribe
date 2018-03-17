import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from models import GeneratorRNN
from models import ConditionedRNN

def generate_sequence(rnn : GeneratorRNN,
        length : int,
        start=[0,0,0],
        bias: int=0):
    """
    Generate a random sequence of handwritten strokes, with `length` strokes.
    """
    if rnn.is_cuda():
        inputs = Variable(torch.Tensor(start).view(1,1,3).float().cuda())
    else:
        inputs = Variable(torch.Tensor(start).view(1,1,3).float())
    hidden = rnn.init_hidden()
    strokes = np.empty([length+1,3])
    strokes[0] = start
    for i in range(1,length+1):
        e,pi,mu,sigma,rho,hidden = rnn.forward(inputs, hidden, bias=bias)

        # Sample from bivariate Gaussian mixture model
        # Choose a component
        pi = pi[0,0,:].view(-1).data.cpu().numpy()
        component = np.random.choice(range(rnn.num_components), p=pi)
        if rnn.is_cuda():
            mu = mu[0,0,component].data.cpu().numpy()
            sigma = sigma[0,0,component].data.cpu().numpy()
            rho = rho[0,0,component].data.cpu().numpy()
        else:
            mu = mu[0,0,component].data.numpy()
            sigma = sigma[0,0,component].data.numpy()
            rho = rho[0,0,component].data.numpy()

        # Sample from the selected Gaussian
        covar = [[sigma[0]**2, rho[0]*sigma[0]*sigma[1]],
                [rho[0]*sigma[0]*sigma[1], sigma[1]**2]] # See https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        sample = np.random.multivariate_normal(mu,covar)

        # Sample from Bernoulli
        if rnn.is_cuda():
            e = e.data.cpu().numpy()
        else:
            e = e.data.numpy()
        lift = np.random.binomial(1,e)[0]

        # Store stroke
        strokes[i] = [lift,sample[0],sample[1]]

        # Update next input
        inputs.data[0][0][0] = int(lift)
        inputs.data[0][0][1] = sample[0]
        inputs.data[0][0][2] = sample[1]

    return strokes

def generate_conditioned_sequence(rnn : ConditionedRNN, length : int,
        sentence, start = [0,0,0], bias: int = 0):
    """
    Generate a sequence of handwritten strokes representing the given sentence, with at most `length` strokes.
    """
    if rnn.is_cuda():
        inputs = Variable(torch.Tensor(start).view(1,1,3).float().cuda())
    else:
        inputs = Variable(torch.Tensor(start).view(1,1,3).float())
    hidden = rnn.init_hidden()
    strokes = np.empty([length+1,3])
    strokes[0] = start
    terminal = False
    sentence = sentence.view(1,sentence.size()[0],sentence.size()[1])
    for i in range(1,length+1):
        e,pi,mu,sigma,rho,hidden,terminal = rnn.forward(inputs, hidden, sentence, bias=bias)
        if terminal:
            strokes = strokes[:i,:]
            break

        # Sample from bivariate Gaussian mixture model
        # Choose a component
        pi = pi[0,0,:].view(-1).data.cpu().numpy()
        component = np.random.choice(range(rnn.num_components), p=pi)
        if rnn.is_cuda():
            mu = mu[0,0,component].data.cpu().numpy()
            sigma = sigma[0,0,component].data.cpu().numpy()
            rho = rho[0,0,component].data.cpu().numpy()
        else:
            mu = mu[0,0,component].data.numpy()
            sigma = sigma[0,0,component].data.numpy()
            rho = rho[0,0,component].data.numpy()

        # Sample from the selected Gaussian
        covar = [[sigma[0]**2, rho*sigma[0]*sigma[1]],
                [rho*sigma[0]*sigma[1], sigma[1]**2]] # See https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        sample = np.random.multivariate_normal(mu,covar)

        # Sample from Bernoulli
        if rnn.is_cuda():
            e = e.data.cpu().numpy()
        else:
            e = e.data.numpy()
        lift = np.random.binomial(1,e)[0]

        # Store stroke
        strokes[i] = [lift,sample[0],sample[1]]

        # Update next input
        inputs.data[0][0][0] = int(lift)
        inputs.data[0][0][1] = sample[0]
        inputs.data[0][0][2] = sample[1]

    return strokes

