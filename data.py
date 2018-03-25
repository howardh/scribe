import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np

def load_data():
    with open('data/strokes.npy','rb') as f:
        strokes = np.load(f,encoding='bytes')
    return strokes

def load_sentences():
    with open('data/sentences.txt','r') as f:
        lines = [x.strip() for x in f.readlines()]
    return lines

def compute_alphabet(sentences):
    alphabet = set()
    for s in tqdm(sentences):
        alphabet.update(set(s))
    alphabet = list(alphabet)
    alphabet.sort()
    alpha_dict = dict([(v,k) for k,v in enumerate(alphabet)])
    return alphabet, alpha_dict

def sentence_to_vectors(sentence, alphabet_dict):
    n = len(alphabet_dict)
    slen = len(sentence)
    output = np.zeros([slen, n])
    for i,c in enumerate(sentence):
        output[i,alphabet_dict[c]] = 1
    return output

def normalize_strokes(strokes):
    concatenated = np.concatenate(strokes)
    m = np.mean(concatenated, axis=0)
    s = np.std(concatenated, axis=0)
    outputs = []
    for stroke in tqdm(strokes,desc='Normalizing Strokes'):
        output = np.empty(stroke.shape)
        for i in range(len(stroke)):
            output[i][0] = stroke[i][0]
            output[i][1] = (stroke[i][1]-m[1])/s[1]
            output[i][2] = (stroke[i][2]-m[2])/s[2]
        outputs.append(output)
    return outputs, m[1:], s[1:]

def unnormalize_strokes(strokes, m, s):
    output = np.empty(strokes.shape)
    for i in range(len(strokes)):
        output[i][0] = strokes[i][0]
        output[i][1] = strokes[i][1]*s[0]+m[0]
        output[i][2] = strokes[i][2]*s[1]+m[1]
    return output

def batch(strokes):
    batch_size = len(strokes)
    max_len = max([len(s) for s in strokes])
    batched_strokes = np.empty([max_len, batch_size,3])
    for i,s in enumerate(strokes):
        batched_strokes[:len(s),i,:] = s
    mask = np.array([[1]*(len(s)-1)+[0]*(max_len-len(s)) for s in strokes]).transpose()
    return batched_strokes, mask

def batch_min(strokes):
    batch_size = len(strokes)
    max_len = min([len(s) for s in strokes])
    batched_strokes = np.empty([max_len, batch_size,3])
    for i,s in enumerate(strokes):
        batched_strokes[:,i,:] = s[:max_len,:]
    mask = np.array([[1]*max_len for s in strokes]).transpose()
    return batched_strokes, mask

def batch_sentences(sentences):
    batch_size = len(sentences)
    max_len = max([len(s) for s in sentences])
    num_chars = sentences[0].size()[1]
    batched_sentences = np.zeros([batch_size,max_len,num_chars])
    for i,s in enumerate(sentences):
        batched_sentences[i,:len(s),:] = s
    return torch.from_numpy(batched_sentences).float()

