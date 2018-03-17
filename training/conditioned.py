import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm

from . import prob
from generator import generate_conditioned_sequence
from models import ConditionedRNN

def compute_loss(rnn, strokes, sentence, mask=None):
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
        e,pi,mu,sigma,rho,hidden,_ = rnn(x, hidden, sentence)
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

def train(rnn : ConditionedRNN, optimizer : torch.optim.Optimizer,
        strokes, sentence):
    total_loss = compute_loss(rnn, strokes, sentence)

    optimizer.zero_grad()
    tqdm.write("Loss: %s" % (total_loss[0].data[0]/len(strokes)))
    total_loss[0].backward()
    torch.nn.utils.clip_grad.clip_grad_norm(rnn.parameters(), 10)
    optimizer.step()

def train_batch(rnn : ConditionedRNN, optimizer : torch.optim.Optimizer, strokes, sentences):
    batched_strokes, mask = batch(strokes)
    batched_sentences = batch_sentences(sentences)
    if rnn.is_cuda():
        mask = Variable(torch.from_numpy(mask).byte().cuda(), requires_grad=False)
        batched_sentences = Variable(batched_sentences.cuda(), requires_grad=False)
    else:
        mask = Variable(torch.from_numpy(mask).byte(), requires_grad=False)
        batched_sentences = Variable(batched_sentences, requires_grad=False)
    total_loss = compute_loss(rnn, batched_strokes, batched_sentences, mask)

    optimizer.zero_grad()
    if type(total_loss[0]) is not int:
        tqdm.write("Loss: %s" % (total_loss[0].data[0]/len(strokes)))
        total_loss[0].backward(retain_graph=True)
        g = print_avg_grad(rnn)
        if g != g:
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
        if g != g:
            tqdm.write("Skipping batch tail")
            return
        torch.nn.utils.clip_grad.clip_grad_norm(rnn.parameters(), 10)
        print_avg_grad(rnn)
        optimizer.step()
    else:
        tqdm.write("No tail")
        return

def train_all(rnn : ConditionedRNN, optimizer : torch.optim.Optimizer, data, sentences, sm, ss, target_sentence=None):
    i = 0
    while True:
        for strokes, sentence in tqdm(zip(data, sentences)):
            if target_sentence is not None and i%50 == 0:
                for b in [0,0.1,5,10]:
                    generated_strokes = generate_conditioned_sequence(rnn, 700,
                            target_sentence, [0,sm[1],sm[2]], bias=b)
                    generated_strokes = unnormalize_strokes(generated_strokes, sm, ss)
                    file_name = 'output_cond/%d-%s.png'%(i,b)
                    plot_stroke(generated_strokes, file_name)
                    tqdm.write('Writing file: %s' % file_name)
                    torch.save(rnn.state_dict(), "models_cond/%d.pt"%i)
            i+=1
            train(rnn, optimizer, strokes, sentence)
    return

def train_all_random_batch(rnn : ConditionedRNN, optimizer : torch.optim.Optimizer,
        data, sentences, sm, ss, target_sentence=None):
    batch_size=10
    i = 0
    pbar = tqdm()
    while True:
        index = np.random.choice(range(len(data)-batch_size),size=1)[0]
        batched_data = data[index:(index+batch_size)]
        batched_sentences = sentences[index:(index+batch_size)]
        #sampled_indices = np.random.choice(range(len(sentences)),size=batch_size,replace=False)
        #batched_data = [data[i] for i in sampled_indices]
        #batched_sentences = [sentences[i] for i in sampled_indices]
        if i%50 == 0:
            for b in [0,0.1,1,5]:
                generated_strokes = generate_conditioned_sequence(rnn, 700,
                        target_sentence, [0,sm[1],sm[2]], bias=b)
                generated_strokes = unnormalize_strokes(generated_strokes, sm, ss)
                file_name = 'output_cond_batch/%d-%s.png'%(i,b)
                plot_stroke(generated_strokes, file_name)
                tqdm.write('Writing file: %s' % file_name)
            torch.save(rnn.state_dict(), "models_cond_batch/%d.pt"%i)
        i+=1
        train_batch(rnn, optimizer, batched_data, batched_sentences)
        pbar.update(1)
    return

