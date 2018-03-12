import numpy as np

from script import compute_alphabet
from script import sentence_to_vectors

def test_compute_alphabet():
    sentence = 'abc'
    alpha, alpha_dict = compute_alphabet(sentence)
    assert 'a' in alpha
    assert 'b' in alpha
    assert 'c' in alpha
    assert 'a' in alpha_dict
    assert 'b' in alpha_dict
    assert 'c' in alpha_dict

def test_sentence_to_vector():
    d = {'a': 0, 'b': 1, 'c': 2}
    sentence = 'abc'
    output = sentence_to_vectors(sentence, d)
    expected_output = np.array([[1,0,0],[0,1,0],[0,0,1]])
    assert (output==expected_output).all()
