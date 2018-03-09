import numpy as np
import torch
from torch.autograd import Variable

from script import GeneratorRNN
from script import prob
from script import train

def test_prob():
    rnn = GeneratorRNN(1)
    inputs = Variable(torch.zeros(1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,hidden = rnn(inputs, hidden)
    p = prob(inputs, (e,pi,mu,sigma,rho))

def test_train():
    rnn = GeneratorRNN(1)
    strokes = np.array([[0,0,0],[0,1,1],[1,2,2]])
    train(rnn, strokes)
