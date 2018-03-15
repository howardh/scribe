import numpy as np
import torch
from torch.autograd import Variable

from script import BivariateGaussianMixtureLayer

def test_split_outputs_all_unique():
    """
    Check that split_outputs() separates everything properly and no output is
    accidentally used for multiple purposes.
    """
    bgm= BivariateGaussianMixtureLayer(1)
    outputs = Variable(torch.from_numpy(np.array(range(7))).view(1,-1,7))
    split = bgm.split_outputs(outputs)
    all_vals = []
    for s in split:
        all_vals += s.data.view(-1).numpy().tolist()
    assert len(all_vals)==7
    all_vals = set(all_vals)
    assert len(all_vals)==7

    bgm = BivariateGaussianMixtureLayer(2)
    outputs = Variable(torch.from_numpy(np.array(range(1+6*2))).view(1,-1,1+6*2))
    split = bgm.split_outputs(outputs)
    all_vals = []
    for s in split:
        all_vals += s.data.view(-1).numpy().tolist()
    assert len(all_vals)==1+6*2
    all_vals = set(all_vals)
    assert len(all_vals)==1+6*2

def test_1comp():
    bgm = BivariateGaussianMixtureLayer(1)
    inputs = Variable(torch.arange(7).view(1,1,7))
    e,pi,mu,sigma,rho = bgm(inputs)

    assert (e==1/2).all()
    assert (pi.size()[1] == 1)
    assert (pi[0,0,0]==1).all()
    assert (mu[0,0,0,0]==2).all(), "%s != 2"%mu[0,0,0,0]
    assert (mu[0,0,0,1]==3).all()
    assert (sigma[0,0,0,0]==np.exp(4)).all()
    assert (sigma[0,0,0,1]==np.exp(5)).all()
    assert (rho[0,0,0]==np.tanh(6)).all()

def test_2comp():
    bgm = BivariateGaussianMixtureLayer(2)
    inputs = Variable(torch.arange(1+6*2).view(1,1,1+6*2))
    e,pi,mu,sigma,rho = bgm(inputs)

    assert (e==1/2).all()

    assert (pi.size()[2] == 2)
    expected_pi = np.exp(1)/(np.exp(1)+np.exp(2))
    assert (pi[0,0,0]==expected_pi).all(), "%s != %s"%(pi[0,0,0],expected_pi)
    expected_pi = np.exp(2)/(np.exp(1)+np.exp(2))
    assert (pi[0,0,1]==expected_pi).all(), "%s != %s"%(pi[0,0,1],expected_pi)

    assert (mu[0,0,0,0]==3).all(), "%s != 3"%mu[0,0,0,0]
    assert (mu[0,0,0,1]==4).all(), "%s != 4"%mu[0,0,0,1]
    assert (mu[0,0,1,0]==5).all(), "%s != 5"%mu[0,0,1,0]
    assert (mu[0,0,1,1]==6).all(), "%s != 6"%mu[0,0,1,1]

    assert (sigma[0,0,0,0]==np.exp(7)).all()
    assert (sigma[0,0,0,1]==np.exp(8)).all()
    assert (sigma[0,0,1,0]==np.exp(9)).all()
    assert (sigma[0,0,1,1]==np.exp(10)).all()

    assert (rho[0,0,0]==np.tanh(11)).all()
    assert (rho[0,0,1]==np.tanh(12)).all()
