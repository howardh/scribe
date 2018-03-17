import numpy as np
import torch
from torch.autograd import Variable

from script import ConditionedRNN
from script import WindowLayer

def test_window_forward_no_errors():
    window = WindowLayer(10,10,1,3)
    seq = Variable(torch.zeros([1,10,3]))
    inputs = Variable(torch.zeros([1,1,10]))
    hidden = window.init_hidden()
    window(inputs, hidden, seq)

    window = WindowLayer(10,10,2,3)
    seq = Variable(torch.zeros([1,10,3]))
    inputs = Variable(torch.zeros([1,1,10]))
    hidden = window.init_hidden()
    window(inputs, hidden, seq)

def test_window_forward_batch():
    text_len = 10
    batch_size = 2
    window = WindowLayer(10,10,7,3)
    text = Variable(torch.rand([batch_size,text_len,3]))
    text1 = text[0,:,:].view(1,text_len,3)
    text2 = text[1,:,:].view(1,text_len,3)
    inputs = Variable(torch.zeros([1,batch_size,10]))
    inputs1 = inputs[:,0,:].view(1,1,10)
    inputs2 = inputs[:,1,:].view(1,1,10)

    hidden = window.init_hidden(batch_size)
    output = window(inputs, hidden, text)

    hidden1 = window.init_hidden()
    output1 = window(inputs1, hidden1, text1)

    hidden2 = window.init_hidden()
    output2 = window(inputs2, hidden2, text2)

    expected_output = torch.cat([output1[0],output2[0]],dim=1)

    diff = torch.abs(expected_output-output[0])
    assert (diff<0.0000001).all()

def test_forward_no_errors():
    """
    Check that the forward pass works without error
    """
    seq = Variable(torch.zeros([1,10,3]))
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
    n_components = 1
    alpha_size = 3
    rnn = ConditionedRNN(n_components,alpha_size)
    seq = Variable(torch.zeros([1,10,3]))
    inputs = Variable(torch.zeros(1,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,_,_ = rnn(inputs, hidden, seq)

    assert (e >= 0).all()
    assert (e <= 1).all()

    diff = torch.abs(1-torch.sum(pi))
    assert (diff < 0.00001).all()

    assert (sigma > 0).all()

    assert (rho > -1).all()
    assert (rho < 1).all()

    rnn = ConditionedRNN(3,3)
    seq = Variable(torch.zeros([1,10,3]))
    inputs = Variable(torch.zeros(1,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,_,_ = rnn(inputs, hidden, seq)

    pi_sum = torch.sum(pi,dim=2)
    diff = torch.abs(1-pi_sum)
    assert (diff < 0.00001).all()

def test_forward():
    rnn = ConditionedRNN(1,3)
    seq = Variable(torch.zeros([1,10,3]))
    inputs = Variable(torch.zeros(1,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,_,_ = rnn(inputs, hidden, seq)

    rnn = ConditionedRNN(3,3)
    seq = Variable(torch.zeros([1,10,3]))
    inputs = Variable(torch.zeros(1,1,3))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,_,_ = rnn(inputs, hidden, seq)

def test_forward_batch():
    seq_len = 1
    batch_len = 2
    features = 3
    text_len = 10
    alpha_size = 3

    rnn = ConditionedRNN(1,alpha_size)

    text = Variable(torch.rand([batch_len,text_len,alpha_size]))
    inputs = Variable(torch.rand(seq_len, batch_len, features))

    text1 = text[0,:,:].view(1,text_len,alpha_size)
    inputs1 = inputs[:,0,:].view(seq_len,1,features)

    text2 = text[1,:,:].view(1,text_len,alpha_size)
    inputs2 = inputs[:,1,:].view(seq_len,1,features)

    hidden = rnn.init_hidden(batch_len)
    e,pi,mu,sigma,rho,_,_ = rnn(inputs, hidden, text)

    hidden = rnn.init_hidden()
    e1,pi1,mu1,sigma1,rho1,_,_ = rnn(inputs1, hidden, text1)

    hidden = rnn.init_hidden()
    e2,pi2,mu2,sigma2,rho2,_,_ = rnn(inputs2, hidden, text2)

    expected_e = torch.cat([e1,e2],1)
    diff = torch.abs(e-expected_e)
    assert (diff < 0.00001).all()

    expected_mu = torch.cat([mu1,mu2],1)
    diff = torch.abs(mu-expected_mu)
    assert (diff < 0.00001).all()
    
    expected_sigma = torch.cat([sigma1,sigma2],1)
    diff = torch.abs(sigma-expected_sigma)
    assert (diff < 0.00001).all()

    expected_rho = torch.cat([rho1,rho2],1)
    diff = torch.abs(rho-expected_rho)
    assert (diff < 0.00001).all()
