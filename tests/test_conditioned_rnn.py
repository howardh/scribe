import numpy as np
import torch
from torch.autograd import Variable

from script import ConditionedRNN
from script import WindowLayer

def test_window_forward_no_errors():
    window = WindowLayer(10,10,1,3)
    seq = Variable(torch.zeros([10,3]))
    inputs = Variable(torch.zeros([1,1,10]))
    hidden = window.init_hidden()
    window(inputs, hidden, seq)

    window = WindowLayer(10,10,2,3)
    seq = Variable(torch.zeros([10,3]))
    inputs = Variable(torch.zeros([1,1,10]))
    hidden = window.init_hidden()
    window(inputs, hidden, seq)

def test_forward_no_errors():
    """
    Check that the forward pass works without error
    """
    seq = Variable(torch.zeros([10,3]))
    rnn = ConditionedRNN(1,3)
    inputs = Variable(torch.zeros(1,1,3))
    hidden = rnn.init_hidden()
    rnn(inputs, hidden, seq)

    rnn = ConditionedRNN(20,3)
    inputs = Variable(torch.zeros(1,1,3))
    hidden = rnn.init_hidden()
    rnn(inputs, hidden, seq)

def test_forward_values():
    """
    Check that all outputs from the forward pass are in the correct range.
    See (18)-(21)
    """
    rnn = ConditionedRNN(1,3)
    seq = Variable(torch.zeros([10,3]))
    inputs = Variable(torch.zeros(1,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,_ = rnn(inputs, hidden, seq)

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

    rnn = ConditionedRNN(3,3)
    seq = Variable(torch.zeros([10,3]))
    inputs = Variable(torch.zeros(10,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,_ = rnn(inputs, hidden, seq)

    pi = pi.data.numpy()
    pi_sum = np.sum(pi,axis=1)
    diff = np.sum(np.abs(1-pi_sum))
    assert diff < 0.00001

def test_forward():
    rnn = ConditionedRNN(1,3)
    seq = Variable(torch.zeros([10,3]))
    inputs = Variable(torch.zeros(1,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,_ = rnn(inputs, hidden, seq)

    rnn = ConditionedRNN(3,3)
    seq = Variable(torch.zeros([10,3]))
    inputs = Variable(torch.zeros(10,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,_ = rnn(inputs, hidden, seq)

def test_forward_batch():
    rnn = ConditionedRNN(1,3)
    seq = Variable(torch.zeros([10,3]))
    seq_len = 1
    batch_len = 2
    features = 3
    inputs = Variable(torch.zeros(seq_len, batch_len, features))
    hidden = rnn.init_hidden()
    e1,pi1,mu1,sigma1,rho1,_ = rnn(inputs, hidden, seq)

    inputs = Variable(torch.zeros(seq_len, 1, features))
    hidden = rnn.init_hidden()
    e2,pi2,mu2,sigma2,rho2,_ = rnn(inputs, hidden, seq)

    diff = e1-e2
    diff = np.sum(np.abs(diff.data.numpy()))
    assert diff < 0.00001

    diff = mu1-mu2
    diff = np.sum(np.abs(diff.data.numpy()))
    assert diff < 0.00001
    
    diff = sigma1-sigma2
    diff = np.sum(np.abs(diff.data.numpy()))
    assert diff < 0.00001

    diff = rho1-rho2
    diff = np.sum(np.abs(diff.data.numpy()))
    assert diff < 0.00001
