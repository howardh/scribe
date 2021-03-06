import numpy as np
import torch
from torch.autograd import Variable

from models import GeneratorRNN
import training
from training import prob
from training import normal
from training import MaskFunction
from training import create_optimizer
from training.unconditioned import train
from training.unconditioned import compute_loss
from training.unconditioned import train_batch
from data import batch

def test_mask():
    x = Variable(torch.ones(3,3).float())
    mask = Variable(torch.ones(3,3).byte())
    f = MaskFunction.apply
    mask.data[0,0] = 0
    expected_output = torch.Tensor([[0,1,1],[1,1,1],[1,1,1]]).float()
    output = f(x,mask)
    output = output.data
    assert (expected_output==output).all()

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
    e =     Variable(torch.Tensor([[0]]).float())
    pi =    Variable(torch.Tensor([[[0.5,0.5]]]).float())
    mu =    Variable(torch.Tensor([[[[0,0],[0,0]]]]).float())
    sigma = Variable(torch.Tensor([[[[1,1],[1,1]]]]).float())
    rho =   Variable(torch.Tensor([[[0,0]]]).float())
    x =     Variable(torch.zeros([1,1,3]).float())

    p = prob(x,(e,pi,mu,sigma,rho))
    p = p.data.numpy()
    expected = 1/(2*np.pi)
    diff = np.abs(p-expected)
    assert diff<0.00001

    x =     Variable(torch.Tensor([[[0,1,2]]]).float())
    e =     Variable(torch.Tensor([[0]]).float())
    pi =    Variable(torch.Tensor([[[0.5,0.5]]]).float())
    mu =    Variable(torch.Tensor([[[[3,4],[5,6]]]]).float())
    sigma = Variable(torch.Tensor([[[[1,2],[3,4]]]]).float())
    rho =   Variable(torch.Tensor([[[0,0]]]).float())

    p = prob(x,(e,pi,mu,sigma,rho))
    p = p.data.numpy()
    z1 = (1-3)**2/1 + (2-4)**2/4
    z2 = (1-5)**2/9 + (2-6)**2/16
    n1 = 1/(2*np.pi*1*2)*np.exp(-z1/2)
    n2 = 1/(2*np.pi*3*4)*np.exp(-z2/2)
    expected = (n1+n2)/2
    diff = np.abs(p-expected)
    assert diff<0.00001

def test_prob_value_sequence_no_error():
    e =     Variable(torch.Tensor([[0]]).float())
    pi =    Variable(torch.Tensor([[[0.5,0.5]]]).float())
    mu =    Variable(torch.Tensor([[[[0,0],[0,0]]]]).float())
    sigma = Variable(torch.Tensor([[[[1,1],[1,1]]]]).float())
    rho =   Variable(torch.Tensor([[[0,0]]]).float())
    x =     Variable(torch.zeros([5,1,3]).float())

    p = prob(x,(e,pi,mu,sigma,rho))

def test_prob_value_batch():
    e =     Variable(torch.Tensor([[0]]).float())
    pi =    Variable(torch.Tensor([[[0.5,0.5]]]).float())
    mu =    Variable(torch.Tensor([[[[0,0],[0,0]]]]).float())
    sigma = Variable(torch.Tensor([[[[1,1],[1,1]]]]).float())
    rho =   Variable(torch.Tensor([[[0,0]]]).float())
    x =     Variable(torch.zeros([1,2,3]).float())

    p = prob(x,(e,pi,mu,sigma,rho))
    diff = torch.abs(p[0][0]-p[0][1])
    assert (diff<0.000001).all()

def test_prob_value_batch_sequence():
    e =     Variable(torch.Tensor([[0]]).float())
    pi =    Variable(torch.Tensor([[[0.5,0.5]]]).float())
    mu =    Variable(torch.Tensor([[[[0,0],[0,0]]]]).float())
    sigma = Variable(torch.Tensor([[[[1,1],[1,1]]]]).float())
    rho =   Variable(torch.Tensor([[[0,0]]]).float())
    x1 =    Variable(torch.arange(5*3).view([5,1,3]).float()*0.02)
    x2 =    Variable(torch.arange(5*3).view([5,1,3]).float()*0.01)
    x =     Variable(torch.zeros([5,2,3]).float())
    x.data[:,0,:] = x1.data
    x.data[:,1,:] = x2.data

    p1 = prob(x1,(e,pi,mu,sigma,rho))
    p2 = prob(x2,(e,pi,mu,sigma,rho))
    p = prob(x,(e,pi,mu,sigma,rho))

    expected_p = torch.cat([p1,p2],dim=1)
    diff = torch.abs(p-expected_p)
    assert (diff<0.000001).all()

def test_normal_value():
    x =     Variable(torch.Tensor([[[1,2]]]).float())
    mu =    Variable(torch.Tensor([[[3,4],[5,6]]]).float())
    sigma = Variable(torch.Tensor([[[1,2],[3,4]]]).float())
    rho =   Variable(torch.Tensor([[0,0]]).float())
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
    x =     Variable(torch.Tensor([[[1,2]]]).float())
    mu =    Variable(torch.Tensor([[[3,4],[5,6]]]).float())
    sigma = Variable(torch.Tensor([[[1,2],[3,4]]]).float())
    rho =   Variable(torch.Tensor([[0.3,0.7]]).float())
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
    rnn = GeneratorRNN(1)
    seq_len = 5
    batch_len = 2
    features = 3
    strokes1 = np.random.rand(seq_len, 1, features)
    strokes2 = np.random.rand(seq_len, 1, features)
    strokes1[:,:,0] = np.floor(strokes1[:,:,0]+0.5)
    strokes2[:,:,0] = np.floor(strokes2[:,:,0]+0.5)
    strokes3 = np.zeros([seq_len, batch_len, features])
    strokes3[:,0,:] = strokes1[:,0,:]
    strokes3[:,1,:] = strokes2[:,0,:]
    loss1 = compute_loss(rnn, strokes1)
    loss2 = compute_loss(rnn, strokes2)
    loss3 = compute_loss(rnn, strokes3)
    loss1 = loss1[0]+loss1[1]
    loss2 = loss2[0]+loss2[1]
    loss3 = loss3[0]+loss3[1]
    print(loss1)
    print(loss2)
    print(loss3)
    diff = torch.abs(loss3[0][0]-(loss1+loss2))
    assert (diff<0.00001).all(), ("Difference too large: %s"%diff)

def test_compute_loss_batch_different_length():
    rnn = GeneratorRNN(1)

    seq_len = 5
    seq_len1 = 5
    seq_len2 = 3
    batch_len = 2
    features = 3

    strokes1 = np.random.rand(seq_len1, 1, features)
    strokes2 = np.random.rand(seq_len2, 1, features)
    strokes1[:,:,0] = np.floor(strokes1[:,:,0]+0.5)
    strokes2[:,:,0] = np.floor(strokes2[:,:,0]+0.5)
    strokes3 = np.zeros([seq_len, batch_len, features])
    strokes3[:seq_len1,0,:] = strokes1[:,0,:]
    strokes3[:seq_len2,1,:] = strokes2[:,0,:]
    mask = Variable(torch.Tensor([[1,1,1,1],[1,1,0,0]]).byte().t())
    loss1 = compute_loss(rnn, strokes1)
    loss2 = compute_loss(rnn, strokes2)
    loss3 = compute_loss(rnn, strokes3, mask)
    loss1 = loss1[0]+loss1[1]
    loss2 = loss2[0]+loss2[1]
    loss3 = loss3[0]+loss3[1]
    print(loss1)
    print(loss2)
    print(loss3)
    diff = torch.abs(loss3-(loss1+loss2))
    assert (diff<0.00001).all(), ("Difference too large: %s"%diff)

def test_compute_loss_batch_different_length_long():
    rnn = GeneratorRNN(1)
    optimizer = create_optimizer(rnn)

    max_len = 500
    seq_len = [2]*49+[max_len]
    batch_len = 50
    features = 3

    strokes = [np.random.rand(l, features) for l in seq_len]
    batched_strokes, mask = batch(strokes)
    train_batch(rnn, optimizer, strokes, batched_strokes, mask)
    assert False

def test_train():
    rnn = GeneratorRNN(1)
    opt = torch.optim.SGD(params=rnn.parameters(),lr=0.0001)
    strokes = np.array([[[0,0,0]],[[0,1,1]],[[1,2,2]]])
    train(rnn, opt, strokes)

def test_batch():
    s1 = np.random.rand(10,3)
    s2 = np.random.rand(11,3)
    s3 = np.random.rand(12,3)
    b,m = batch([s1,s2,s3])
    expected_mask = np.array([[1,1,1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1,1,1,1]]).transpose()
    print(m)
    print(expected_mask)
    print(m==expected_mask)
    assert (m==expected_mask).all()
