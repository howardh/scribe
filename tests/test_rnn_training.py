import numpy as np
import torch
from torch.autograd import Variable

from script import GeneratorRNN
from script import prob
from script import train

def test_prob_size():
    rnn = GeneratorRNN(1)
    inputs = Variable(torch.zeros(1,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,hidden = rnn(inputs, hidden)
    p = prob(inputs, (e,pi,mu,sigma,rho))
    assert np.prod(p.size()) == 1

    rnn = GeneratorRNN(20)
    inputs = Variable(torch.zeros(1,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,hidden = rnn(inputs, hidden)
    p = prob(inputs, (e,pi,mu,sigma,rho))
    assert np.prod(p.size()) == 1

def test_prob_value():
    e =     Variable(torch.Tensor([0]).float())
    pi =    Variable(torch.Tensor([[0.5,0.5]]).float())
    mu =    Variable(torch.Tensor([[[0,0],[0,0]]]).float())
    sigma = Variable(torch.Tensor([[[1,1],[1,1]]]).float())
    rho =   Variable(torch.Tensor([[0,0]]).float())
    x =     Variable(torch.Tensor([[[0,0,0]]]).float())

    p = prob(x,(e,pi,mu,sigma,rho))
    p = p.data.numpy()
    expected = 1/(2*np.pi)
    diff = np.abs(p-expected)
    assert diff<0.00001

def test_train():
    rnn = GeneratorRNN(1)
    opt = torch.optim.SGD(params=rnn.parameters(),lr=0.0001)
    strokes = np.array([[[0,0,0]],[[0,1,1]],[[1,2,2]]])
    train(rnn, opt, strokes)
