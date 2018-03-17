import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np
from utils import plot_stroke

from models import GeneratorRNN
from models import ConditionedRNN
from training import create_optimizer
from training.unconditioned import train_all_random_batch

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

if __name__=='__main__':
    data = load_data()
    sorted_indices = np.argsort([len(d) for d in data])

    data = [data[i] for i in sorted_indices]
    normalized_data,m,s = normalize_strokes(data)

    sentences = load_sentences()
    sentences = [sentences[i] for i in sorted_indices]
    alphabet, alphabet_dict = compute_alphabet(sentences)
    sentence_vars = [Variable(torch.from_numpy(sentence_to_vectors(s,alphabet_dict)).float().cuda(),
        requires_grad=False) for s in tqdm(sentences,desc="Converting Sentences")]

    rnn = GeneratorRNN(20).cuda()
    #rnn.lstm.register_backward_hook(printgradnorm)
    #rnn.linear.register_backward_hook(printgradnorm)
    #rnn.bgm.register_backward_hook(printgradnorm)
    #rnn = ConditionedRNN(20, len(alphabet)).cuda()
    #rnn.load_state_dict(torch.load('model-all.pt'))
    #rnn.load_state_dict(torch.load('models_cond/1550.pt'))

    optimizer = create_optimizer(rnn)

    #train_all(rnn, optimizer, normalized_data, m.tolist(), s.tolist())
    #train_one(rnn, optimizer, normalized_data[0], m.tolist(), s.tolist())
    #train(rnn, optimizer, normalized_data[0])
    train_all_random_batch(rnn, optimizer, normalized_data, m.tolist(), s.tolist())

    #target_sentence = Variable(torch.from_numpy(sentence_to_vectors("Hello World!",alphabet_dict)).float().cuda(), requires_grad=False)
    #train_all_conditioned(rnn, optimizer, normalized_data, sentence_vars, m.tolist(),
    #        s.tolist(), target_sentence=target_sentence)
    #train_conditioned(rnn, optimizer, normalized_data[0], sentence_vars[0])
    #train_all_random_batch_conditioned(rnn, optimizer, normalized_data, sentence_vars, m.tolist(),
    #        s.tolist(), target_sentence=target_sentence)
