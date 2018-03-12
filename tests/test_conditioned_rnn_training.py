import numpy as np
import torch
from torch.autograd import Variable

from script import ConditionedRNN
from script import train_conditioned
from script import sentence_to_vectors

def test_compute_loss():
    pass

def test_compute_loss_batch():
    pass

def test_train():
    d = {'a': 0, 'b': 1, 'c': 2}
    sentence_vec = sentence_to_vectors('abc',d)
    sentence_vec = Variable(torch.from_numpy(sentence_vec).float(), requires_grad=False)
    rnn = ConditionedRNN(1, len(d))
    opt = torch.optim.SGD(params=rnn.parameters(),lr=0.0001)
    strokes = np.array([[[0,0,0]],[[0,1,1]],[[1,2,2]]])
    train_conditioned(rnn, opt, strokes, sentence_vec)
