import numpy as np
import torch
from torch.autograd import Variable

from script import GeneratorRNN

def test_forward_no_errors():
    """
    Check that the forward pass works without error
    """
    rnn = GeneratorRNN(1)
    inputs = Variable(torch.zeros(1,1,3))
    hidden = rnn.init_hidden()
    rnn(inputs, hidden)

    rnn = GeneratorRNN(20)
    inputs = Variable(torch.zeros(1,1,3))
    hidden = rnn.init_hidden()
    rnn(inputs, hidden)

def test_forward_values():
    """
    Check that all outputs from the forward pass are in the correct range.
    See (18)-(21)
    """
    rnn = GeneratorRNN(1)
    inputs = Variable(torch.zeros(1,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,_ = rnn(inputs, hidden)

    assert (e >= 0).all()
    assert (e <= 1).all()

    diff = torch.abs(1-torch.sum(pi))
    assert (diff < 0.00001).all()

    assert (sigma > 0).all()

    assert (rho > -1).all()
    assert (rho < 1).all()

    rnn = GeneratorRNN(3)
    inputs = Variable(torch.zeros(10,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,_ = rnn(inputs, hidden)

    pi_sum = torch.sum(pi,dim=2)
    diff = torch.abs(1-pi_sum)
    assert (diff < 0.00001).all()

def test_forward():
    rnn = GeneratorRNN(1)
    inputs = Variable(torch.zeros(1,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,_ = rnn(inputs, hidden)

    rnn = GeneratorRNN(3)
    inputs = Variable(torch.zeros(10,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,_ = rnn(inputs, hidden)

def test_forward_batch():
    rnn = GeneratorRNN(7)
    seq_len = 5
    batch_len = 2
    features = 3
    strokes1 = Variable(torch.arange(seq_len*features).view(seq_len, 1, features))
    strokes2 = Variable(torch.arange(seq_len*features).view(seq_len, 1, features))
    strokes3 = Variable(torch.zeros([seq_len, batch_len, features]))
    strokes3.data[:,0,:] = strokes1.data[:,0,:]
    strokes3.data[:,1,:] = strokes1.data[:,0,:]

    hidden = rnn.init_hidden(2)
    e2,pi2,mu2,sigma2,rho2,_ = rnn(strokes3, hidden)

    hidden = rnn.init_hidden(1)
    e1,pi1,mu1,sigma1,rho1,_ = rnn(strokes1, hidden)

    diff = torch.abs(e1-e2)
    assert (diff < 0.000001).all()
    diff = torch.abs(pi1-pi2)
    assert (diff < 0.000001).all()
    diff = torch.abs(mu1-mu2)
    assert (diff < 0.000001).all()
    diff = torch.abs(sigma1-sigma2)
    assert (diff < 0.000001).all()
    diff = torch.abs(rho1-rho2)
    assert (diff < 0.000001).all()

