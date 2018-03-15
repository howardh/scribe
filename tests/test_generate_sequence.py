import numpy as np
import torch
from torch.autograd import Variable
from unittest.mock import MagicMock

from script import sentence_to_vectors
from script import GeneratorRNN
from script import generate_sequence
from script import ConditionedRNN
from script import generate_conditioned_sequence
from script import compute_loss
from script import BivariateGaussianMixtureLayer
from utils import plot_stroke

def test_no_errors():
    rnn = GeneratorRNN(1)
    strokes = generate_sequence(rnn, 10)
    plot_stroke(strokes, 'strokes.png')

    rnn = GeneratorRNN(20)
    strokes = generate_sequence(rnn, 10)
    plot_stroke(strokes, 'strokes.png')

    rnn = GeneratorRNN(20)
    strokes = generate_sequence(rnn, 10, bias=10)
    plot_stroke(strokes, 'strokes.png')

def test_conditioned_no_errors():
    d = {'a': 0, 'b': 1, 'c': 2}
    sentence_vec = sentence_to_vectors('abcabc',d)
    sentence_vec = Variable(torch.from_numpy(sentence_vec).float(), requires_grad=False)

    rnn = ConditionedRNN(1,3)
    strokes = generate_conditioned_sequence(rnn, 10, sentence_vec)
    plot_stroke(strokes, 'strokes.png')

    rnn = ConditionedRNN(20,3)
    strokes = generate_conditioned_sequence(rnn, 10, sentence_vec)
    plot_stroke(strokes, 'strokes.png')

    rnn = ConditionedRNN(20,3)
    strokes = generate_conditioned_sequence(rnn, 10, sentence_vec, bias=10)
    plot_stroke(strokes, 'strokes.png')

def test_biased_generation():
    rnn = GeneratorRNN(20)
    strokes1 = generate_sequence(rnn, 20, bias=10000)
    strokes2 = generate_sequence(rnn, 20, bias=10000)
    diff = np.abs(strokes1-strokes2)
    print(diff)
    #assert np.sum(diff)==0

def test_correct_distribution():
    """
    Check that the loss function matches the sequence generation function.
    No assertions are made here. Must check manually.
    """
    rnn1 = GeneratorRNN(1)
    rnn2 = GeneratorRNN(1)
    strokes1 = generate_sequence(rnn1, 20, bias=10000)
    strokes2 = generate_sequence(rnn2, 20, bias=10000)
    loss11 = compute_loss(rnn1, strokes1)
    loss22 = compute_loss(rnn2, strokes2)
    loss12 = compute_loss(rnn1, strokes2)
    loss21 = compute_loss(rnn2, strokes1)
    print("loss11", loss11)
    print("loss21", loss21)
    print("loss22", loss22)
    print("loss12", loss12)
    #assert (loss12>loss11).all()
    #assert (loss21>loss22).all()
    #assert False

def test_fixed_distribution():
    bgm = BivariateGaussianMixtureLayer(1)
    inputs = Variable(torch.arange(7).view(1,1,7))
    bgm_output = bgm(inputs)

    rnn = GeneratorRNN(1)
    rnn.bgm.forward = MagicMock(return_value=bgm_output)
    inputs = Variable(torch.zeros([1,1,3]))
    hidden = rnn.init_hidden()
    e,pi,mu,sigma,rho,hidden = rnn(inputs,hidden)
    print(e,pi,mu,sigma,rho)
    #assert False
