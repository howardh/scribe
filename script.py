import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np
from utils import plot_stroke
import argparse

from data import load_data
from data import normalize_strokes
from data import load_sentences
from data import compute_alphabet
from data import sentence_to_vectors
from models import GeneratorRNN
from models import ConditionedRNN
from training import create_optimizer
from training import conditioned
from training import unconditioned
import generator

def print_avg_grad(rnn):
    grads = []
    vals = []
    for p in rnn.parameters():
        grads+= list(np.abs(p.grad.view(-1).data.cpu().numpy()))
        vals += list(np.abs(p.view(-1).data.cpu().numpy()))
    tqdm.write('Average grad: %f, Average vals: %f' % (np.mean(grads), np.mean(vals)))
    return np.mean(grads)

def printgradnorm(self, grad_input, grad_output):
    gin = grad_input[0].data.norm()
    #gout = grad_output[0].data.norm()
    if gin != gin:
        print('Inside ' + self.__class__.__name__ + ' backward')
        print('Inside class:' + self.__class__.__name__)
        print('')
        print('grad_input: ', type(grad_input))
        print('len(grad_input): ', len(grad_input))
        print('grad_input[0]: ', type(grad_input[0]))
        print('grad_output: ', type(grad_output))
        print('len(grad_output): ', len(grad_output))
        print('grad_output[0]: ', type(grad_output[0]))
        print('')
        print('grad_input size:', grad_input[0].size())
        print('grad_output size:', grad_output[0].size())
        print('grad_input norm:', grad_input[0].data.norm())
        for i,g in enumerate(grad_input):
            print("%d %s" % (i,g))
        raise Exception("NAN!")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Handwriting generation')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--conditioned', action='store_true')
    group.add_argument('--unconditioned', action='store_true')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--generate', action='store_true')

    group = parser.add_argument_group('Training:')
    group.add_argument('--batch', action='store_true')
    group.add_argument('--output-dir', type=str)

    parser.add_argument('--output-text', type=str)
    parser.add_argument('--model', type=str)

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_arguments()

    data = load_data()
    sorted_indices = np.argsort([len(d) for d in data])

    data = [data[i] for i in sorted_indices]
    normalized_data,m,s = normalize_strokes(data)

    sentences = load_sentences()
    sentences = [sentences[i] for i in sorted_indices]
    alphabet, alphabet_dict = compute_alphabet(sentences)
    sentence_vars = [Variable(torch.from_numpy(sentence_to_vectors(s,alphabet_dict)).float().cuda(),
        requires_grad=False) for s in tqdm(sentences,desc="Converting Sentences")]

    if args.train:
        if args.unconditioned:
            rnn = GeneratorRNN(num_components=20, mean=m.tolist(), std=s.tolist()).cuda()
            if args.model is not None:
                rnn.load_state_dict(torch.load(args.model))
            optimizer = create_optimizer(rnn)
            if args.batch:
                unconditioned.train_all_random_batch(rnn, optimizer, normalized_data)
            else:
                unconditioned.train_all(rnn, optimizer, normalized_data)
        elif args.conditioned:
            rnn = ConditionedRNN(20, len(alphabet)).cuda()
            if args.model is not None:
                rnn.load_state_dict(torch.load(args.model))
            optimizer = create_optimizer(rnn)
            if args.batch:
                conditioned.train_all_random_batch(rnn, optimizer, normalized_data)
            else:
                conditioned.train_all(rnn, optimizer, normalized_data)
    elif args.generate:
        if args.unconditioned:
            rnn = GeneratorRNN(num_components=20, mean=m.tolist(), std=s.tolist()).cuda()
            if args.model is not None:
                rnn.load_state_dict(torch.load(args.model))
            generator.generate_sequence(rnn, 700, 1)
        elif args.conditioned:
            rnn = ConditionedRNN(20, len(alphabet)).cuda()
            if args.model is not None:
                rnn.load_state_dict(torch.load(args.model))
            target_sentence = Variable(torch.from_numpy(sentence_to_vectors("Hello World!",alphabet_dict)).float().cuda(), requires_grad=False)
            generator.generate_conditioned_sequence(rnn, 700, target_sentence, 1)

    #rnn = ConditionedRNN(20, len(alphabet)).cuda()
    #rnn.load_state_dict(torch.load('model-all.pt'))
    #rnn.load_state_dict(torch.load('models_cond/1550.pt'))

    #optimizer = create_optimizer(rnn)
