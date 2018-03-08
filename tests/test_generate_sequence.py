import numpy as np
import torch
from torch.autograd import Variable

from script import GeneratorRNN
from script import generate_sequence
from utils import plot_stroke

def test_no_errors():
    rnn = GeneratorRNN(1)
    strokes = generate_sequence(rnn, 10)
    plot_stroke(strokes, 'strokes.png')
