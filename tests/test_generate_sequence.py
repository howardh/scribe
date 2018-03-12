import numpy as np
import torch
from torch.autograd import Variable

from script import sentence_to_vectors
from script import GeneratorRNN
from script import generate_sequence
from script import ConditionedRNN
from script import generate_conditioned_sequence
from script import compute_loss
from utils import plot_stroke

def test_no_errors():
    rnn = GeneratorRNN(1)
    strokes = generate_sequence(rnn, 10)
    plot_stroke(strokes, 'strokes.png')

    rnn = GeneratorRNN(20)
    strokes = generate_sequence(rnn, 10)
    plot_stroke(strokes, 'strokes.png')

def test_conditioned_no_errors():
    d = {'a': 0, 'b': 1, 'c': 2}
    sentence_vec = sentence_to_vectors('abc',d)
    sentence_vec = Variable(torch.from_numpy(sentence_vec).float(), requires_grad=False)

    rnn = ConditionedRNN(1,3)
    strokes = generate_conditioned_sequence(rnn, 10, sentence_vec)
    plot_stroke(strokes, 'strokes.png')

    rnn = ConditionedRNN(20,3)
    strokes = generate_conditioned_sequence(rnn, 10, sentence_vec)
    plot_stroke(strokes, 'strokes.png')

def test_correct_distribution():
    """
    Check that the loss function matches the sequence generation function.
    No assertions are made here. Must check manually.
    """
    #rnn1 = GeneratorRNN(20)
    #rnn2 = GeneratorRNN(20)
    #strokes1 = generate_sequence(rnn1, 20)
    #strokes2 = generate_sequence(rnn2, 20)
    #loss11 = compute_loss(rnn1, strokes1)
    #loss22 = compute_loss(rnn2, strokes2)
    #loss12 = compute_loss(rnn1, strokes2)
    #loss21 = compute_loss(rnn2, strokes1)
    #print("loss11", loss11)
    #print("loss22", loss22)
    #print("loss12", loss12)
    #print("loss21", loss21)
    pass
