import numpy as np
import torch
from torch.autograd import Variable

from script import GeneratorRNN
from script import prob
from script import normal
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

    x =     Variable(torch.Tensor([[[0,1,2]]]).float())
    e =     Variable(torch.Tensor([0]).float())
    pi =    Variable(torch.Tensor([[0.5,0.5]]).float())
    mu =    Variable(torch.Tensor([[[3,4],[5,6]]]).float())
    sigma = Variable(torch.Tensor([[[1,2],[3,4]]]).float())
    rho =   Variable(torch.Tensor([[0,0]]).float())

    p = prob(x,(e,pi,mu,sigma,rho))
    p = p.data.numpy()
    z1 = (1-3)**2/1 + (2-4)**2/4
    z2 = (1-5)**2/9 + (2-6)**2/16
    n1 = 1/(2*np.pi*1*2)*np.exp(-z1/2)
    n2 = 1/(2*np.pi*3*4)*np.exp(-z2/2)
    expected = (n1+n2)/2
    diff = np.abs(p-expected)
    assert diff<0.00001

def test_normal_value():
    x =     Variable(torch.Tensor([[1,2]]).float())
    mu =    Variable(torch.Tensor([[3,4],[5,6]]).float())
    sigma = Variable(torch.Tensor([[1,2],[3,4]]).float())
    rho =   Variable(torch.Tensor([0,0]).float())
    output = normal(x,mu,sigma,rho)
    output = output.data.numpy()
    z1 = (1-3)**2/1 + (2-4)**2/4
    z2 = (1-5)**2/9 + (2-6)**2/16
    n1 = 1/(2*np.pi*1*2)*np.exp(-z1/2)
    n2 = 1/(2*np.pi*3*4)*np.exp(-z2/2)
    expected = np.array([n1,n2])
    diff = np.sum(np.abs(output-expected))
    assert diff<0.00001

def test_normal_value2():
    x =     Variable(torch.Tensor([[1,2]]).float())
    mu =    Variable(torch.Tensor([[3,4],[5,6]]).float())
    sigma = Variable(torch.Tensor([[1,2],[3,4]]).float())
    rho =   Variable(torch.Tensor([0.3,0.7]).float())
    output = normal(x,mu,sigma,rho)
    output = output.data.numpy()
    z1 = (1-3)**2/1 + (2-4)**2/4 - 2*0.3*(1-3)*(2-4)/(1*2)
    z2 = (1-5)**2/9 + (2-6)**2/16 - 2*0.7*(1-5)*(2-6)/(3*4)
    n1 = 1/(2*np.pi*1*2*np.sqrt(1-0.09))*np.exp(-z1/(2*(1-0.09)))
    n2 = 1/(2*np.pi*3*4*np.sqrt(1-0.49))*np.exp(-z2/(2*(1-0.49)))
    expected = np.array([n1,n2])
    diff = np.sum(np.abs(output-expected))
    assert diff<0.00001

def test_compute_loss():
    pass

def test_compute_loss_batch():
    pass

def test_train():
    rnn = GeneratorRNN(1)
    opt = torch.optim.SGD(params=rnn.parameters(),lr=0.0001)
    strokes = np.array([[[0,0,0]],[[0,1,1]],[[1,2,2]]])
    train(rnn, opt, strokes)
