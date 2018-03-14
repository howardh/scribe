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

    e = e.data.numpy()
    assert e >= 0
    assert e <= 1

    pi = pi.data.numpy()
    pi_sum = np.sum(pi)
    diff = np.abs(1-pi_sum)
    assert diff < 0.00001

    sigma = sigma.view(-1).data.numpy()
    for s in sigma:
        assert s > 0

    rho = rho.data.numpy()
    assert rho > -1
    assert rho < 1

    rnn = GeneratorRNN(3)
    inputs = Variable(torch.zeros(10,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,_ = rnn(inputs, hidden)

    pi = pi.data.numpy()
    pi_sum = np.sum(pi,axis=1)
    diff = np.sum(np.abs(1-pi_sum))
    assert diff < 0.00001

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
    rnn = GeneratorRNN(1)
    seq_len = 1
    batch_len = 2
    features = 3
    inputs = Variable(torch.zeros(seq_len, batch_len, features))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,_ = rnn(inputs, hidden)
    print(mu)

    inputs = Variable(torch.zeros(1,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,_ = rnn(inputs, hidden)
    print(mu)
    #assert False
    #TODO: Assert that the values from the batch and non-batch forward call are the same
