import numpy as np
import torch
from torch.autograd import Variable
from script import load_data
from script import load_sentences
from script import normalize_strokes
from script import unnormalize_strokes
from script import compute_alphabet
from script import GeneratorRNN
from script import generate_sequence
from script import ConditionedRNN
from script import generate_conditioned_sequence
from script import sentence_to_vectors

data = load_data()
normalized_data,m,s = normalize_strokes(data)
sentences = load_sentences()
alphabet, alphabet_dict = compute_alphabet(sentences)

def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    np.random.seed(random_seed)
    rnn = GeneratorRNN(20)
    rnn.load_state_dict(torch.load('model-all.pt'))
    stroke = generate_sequence(rnn, 700)
    stroke = unnormalize_strokes(stroke, m, s)
    return stroke


def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)

    np.random.seed(random_seed)
    rnn = ConditionedRNN(20, len(alphabet))
    rnn.load_state_dict(torch.load('model-cond.pt'))
    text_vec = sentence_to_vectors(text, alphabet_dict)
    text_var = Variable(torch.from_numpy(text_vec).float(), requires_grad=False)
    stroke = generate_conditioned_sequence(rnn, 700, text_var)
    stroke = unnormalize_strokes(stroke, m, s)
    return stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'
