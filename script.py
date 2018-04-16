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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Handwriting generation')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--conditioned', action='store_true')
    group.add_argument('--unconditioned', action='store_true')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--generate', action='store_true')

    group = parser.add_argument_group('Training')
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
    m = m.tolist()
    s = s.tolist()

    sentences = load_sentences()
    sentences = [sentences[i] for i in sorted_indices]
    alphabet, alphabet_dict = compute_alphabet(sentences)
    sentence_vars = [Variable(torch.from_numpy(sentence_to_vectors(s,alphabet_dict)).float().cuda(),
        requires_grad=False) for s in tqdm(sentences,desc="Converting Sentences")]

    # Create RNN
    m = [0.41261038184165955, -0.006002499256283045]
    s = [2.0667049884796143, 1.8475052118301392]
    if args.unconditioned:
        rnn = GeneratorRNN(num_components=20, mean=m, std=s).cuda()
    elif args.conditioned:
        rnn = ConditionedRNN(num_components=20, mean=m, std=s, num_chars=len(alphabet)).cuda()

    # Load weights
    if args.model is not None:
        print("Loading model weights from file %s" % args.model)
        rnn.load_state_dict(torch.load(args.model))

    # Train/Generate
    if args.train:
        optimizer = create_optimizer(rnn)
        if args.unconditioned:
            if args.batch:
                unconditioned.train_all_random_batch(rnn, optimizer, normalized_data)
            else:
                unconditioned.train_all(rnn, optimizer, normalized_data)
        elif args.conditioned:
            if args.batch:
                conditioned.train_all_random_batch(rnn, optimizer, normalized_data)
            else:
                conditioned.train_all(rnn, optimizer, normalized_data)
    elif args.generate:
        if args.unconditioned:
            print("Generating a random handwriting sample.")
            strokes = generator.generate_sequence(rnn, 700, 1)
        elif args.conditioned:
            target_sentence = args.output_text
            print("Generating handwriting for text: %s" % target_sentence)
            target_sentence = Variable(
                    torch.from_numpy(
                        sentence_to_vectors(target_sentence,alphabet_dict)
                    ).float().cuda(),
                    requires_grad=False
            )
            strokes = generator.generate_conditioned_sequence(rnn, 2000,
                    target_sentence, 3)
        plot_stroke(strokes, "output.png")
