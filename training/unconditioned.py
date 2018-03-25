import torch
import torch.nn.functional as F
from torch.autograd import Variable

import os
import numpy as np
from tqdm import tqdm

from . import prob
from data import batch
from data import batch_min
from generator import generate_sequence
from models import GeneratorRNN

def compute_loss(rnn, strokes, mask=None):
    seq_len = strokes.shape[0]
    batch_size = strokes.shape[1]
    if mask is None:
        if rnn.is_cuda():
            mask = Variable(torch.ones(seq_len, batch_size).byte().cuda(), requires_grad=False)
        else:
            mask = Variable(torch.ones(seq_len, batch_size).byte(), requires_grad=False)
    strokes_tensor = torch.from_numpy(strokes).float()
    if rnn.is_cuda():
        strokes_tensor = strokes_tensor.cuda()
    strokes_var = Variable(strokes_tensor, requires_grad=False)
    hidden = rnn.init_hidden(batch_size)
    total_loss = 0
    total_loss2 = 0
    for i in tqdm(range(len(strokes)-1)):
        x = strokes_var[i].view(1,-1,3)
        e,pi,mu,sigma,rho,hidden = rnn(x, hidden)
        y = (e,pi,mu,sigma,rho)
        x2 = strokes_var[i+1].view(1,-1,3)


        likelihood = torch.masked_select(prob(x2,y),mask[i,:])
        likelihood = likelihood[(likelihood > 0.00000001).detach()]
        loss = -torch.log(likelihood)
        if (loss==0).all():
            continue
        if not mask[i,:].all():
            total_loss2 += torch.sum(loss)
        else:
            total_loss += torch.sum(loss)

    return total_loss, total_loss2

def train(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, strokes):
    total_loss = compute_loss(rnn, strokes)
    total_loss = total_loss[0]

    optimizer.zero_grad()
    tqdm.write("Loss: %s" % (total_loss.data[0]/len(strokes)))
    total_loss.backward()
    torch.nn.utils.clip_grad.clip_grad_norm(rnn.parameters(), 10)
    optimizer.step()

def train_truncated_batch(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, strokes):
    batched_strokes, mask = batch_min(strokes)
    train_batch(rnn, optimizer, strokes, batched_strokes, mask)

def train_full_batch(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, strokes):
    batched_strokes, mask = batch(strokes)
    train_batch(rnn, optimizer, strokes, batched_strokes, mask)

def train_batch(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, strokes,
        batched_strokes, mask):
    tqdm.write("Mask: %d"%np.sum(mask==0))
    if rnn.is_cuda():
        mask = Variable(torch.from_numpy(mask).byte().cuda(), requires_grad=False)
    else:
        mask = Variable(torch.from_numpy(mask).byte(), requires_grad=False)
    total_loss = compute_loss(rnn, batched_strokes, mask)

    optimizer.zero_grad()
    if type(total_loss[0]) is not int:
        tqdm.write("Loss: %s" % (total_loss[0].data[0]/len(strokes)))
        total_loss[0].backward(retain_graph=True)
        g = print_avg_grad(rnn)
        if g != g: #FIXME: Hacky
            tqdm.write("Skipping batch")
            return
        torch.nn.utils.clip_grad.clip_grad_norm(rnn.parameters(), 10)
        print_avg_grad(rnn)
        optimizer.step()
    else:
        tqdm.write("Something's wrong.")
        return
    if type(total_loss[1]) is not int:
        optimizer.zero_grad()
        total_loss[1].backward()
        g = print_avg_grad(rnn)
        if g != g: #FIXME: Hacky
            tqdm.write("Skipping batch tail")
            return
        torch.nn.utils.clip_grad.clip_grad_norm(rnn.parameters(), 10)
        print_avg_grad(rnn)
        optimizer.step()
    else:
        tqdm.write("No tail.")
        return

def train_all(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, data):
    i = 0
    while True:
        for strokes in tqdm(data):
            if i%50 == 0:
                for b in [0,0.1,1,5]:
                    generated_strokes = generate_sequence(rnn, 700, bias=b)
                    generated_strokes = unnormalize_strokes(generated_strokes,
                            rnn.mean, rnn.std)
                    file_name = 'output/uncond/%d-%s.png'%(i,b)
                    plot_stroke(generated_strokes, file_name)
                    tqdm.write('Writing file: %s' % file_name)
                torch.save(rnn.state_dict(), "output/models/%d.pt"%i)
            i+=1
            train(rnn, optimizer, strokes)
    return

def train_all_random_batch(rnn : GeneratorRNN, optimizer :
        torch.optim.Optimizer, data, output_directory='./output', tail=True):
    batch_size=100
    i = 0
    pbar = tqdm()
    while True:
        index = np.random.choice(range(len(data)-batch_size),size=1)[0]
        batched_data = data[index:(index+batch_size)]
        if i%50 == 0:
            for b in [0,0.1,1,5]:
                generated_strokes = generate_sequence(rnn, 700, bias=b)
                generated_strokes = unnormalize_strokes(generated_strokes,
                        rnn.mean, rnn.std)
                file_name = os.path.join(output_directory,'batch_uncond/%d-%s.png'%(i,b))
                plot_stroke(generated_strokes, file_name)
                tqdm.write('Writing file: %s' % file_name)
            model_file_name = os.path.join(output_directory, 'models_batch_uncond/%d.pt'%i)
            torch.save(rnn.state_dict(), model_file_name)
        i+=1
        if tail:
            train_full_batch(rnn, optimizer, batched_data)
        else:
            train_truncated_batch(rnn, optimizer, batched_data)
        pbar.update(1)
    return
