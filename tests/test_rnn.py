import numpy as np
import torch
from torch.autograd import Variable

from script import GeneratorRNN

def test_split_outputs_all_unique():
    """
    Check that split_outputs() separates everything properly and no output is
    accidentally used for multiple purposes.
    """
    rnn = GeneratorRNN(1)
    outputs = Variable(torch.from_numpy(np.array(range(7))))
    split = rnn.split_outputs(outputs)
    all_vals = []
    for s in split:
        all_vals += s.data.view(-1).numpy().tolist()
    assert len(all_vals)==7
    all_vals = set(all_vals)
    assert len(all_vals)==7

    rnn = GeneratorRNN(2)
    outputs = Variable(torch.from_numpy(np.array(range(1+6*2))))
    split = rnn.split_outputs(outputs)
    all_vals = []
    for s in split:
        all_vals += s.data.view(-1).numpy().tolist()
    assert len(all_vals)==1+6*2
    all_vals = set(all_vals)
    assert len(all_vals)==1+6*2

def test_forward_no_errors():
    """
    Check that the forward pass works without error
    """
    rnn = GeneratorRNN(1)
    inputs = Variable(torch.zeros(1,3))
    hidden = rnn.init_hidden()
    rnn(inputs, hidden)

    rnn = GeneratorRNN(2)
    inputs = Variable(torch.zeros(1,3))
    hidden = rnn.init_hidden()
    rnn(inputs, hidden)
