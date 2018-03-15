import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np
from utils import plot_stroke

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
    return outputs, m, s

def unnormalize_strokes(strokes, m, s):
    output = np.empty(strokes.shape)
    for i in range(len(strokes)):
        output[i][0] = strokes[i][0]
        output[i][1] = strokes[i][1]*s[1]+m[1]
        output[i][2] = strokes[i][2]*s[2]+m[2]
    return output

class BivariateGaussianMixtureLayer(torch.nn.Module):
    def __init__(self, num_components=1):
        super(BivariateGaussianMixtureLayer,self).__init__()
        # See page 23 for parameters
        self.num_components = num_components
        self.softmax = torch.nn.Softmax(dim=2)

    def split_outputs(self, output):
        e = output[:,:,0]

        start = 1
        end = 1+self.num_components
        pi = output[:,:,start:end]

        start = end
        end = end+2*self.num_components
        mu = output[:,:,start:end]

        start = end
        end = end+2*self.num_components
        sigma = output[:,:,start:end]

        start = end
        end = end+self.num_components
        rho = output[:,:,start:end]

        return e,pi,mu,sigma,rho

    def forward(self, input, bias=0):
        #print('bgm input ', input.size())
        seq_len = input.size()[0]
        # batch size * sequence len * features (3)
        e,pi,mu,sigma,rho = self.split_outputs(input)
        #print('Preprocessing...')
        #print('e ', e.size())
        #print('pi ', pi.size())
        #print('mu ', mu.size())
        #print('sigma ', sigma.size())
        #print('rho ', rho.size())

        n = self.num_components
        e = 1/(1+torch.exp(e))
        pi = self.softmax(pi*(1+bias))
        mu = mu.contiguous().view(seq_len,-1,n,2)
        sigma = torch.exp(sigma-bias).view(seq_len,-1,n,2)
        rho = torch.tanh(rho)
        #print('Postprocessing...')
        #print('e ', e.size())
        #print('pi ', pi.size())
        #print('mu ', mu.size())
        #print('sigma ', sigma.size())
        #print('rho ', rho.size())
        return e,pi,mu,sigma,rho

class GeneratorRNN(torch.nn.Module):
    def __init__(self, num_components=1):
        super(GeneratorRNN,self).__init__()
        # See page 23 for parameters
        self.num_components = num_components
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=900, num_layers=1,
                bias=True)
        self.linear = torch.nn.Linear(in_features=900,out_features=1+(1+2+2+1)*self.num_components)
        self.bgm = BivariateGaussianMixtureLayer(num_components)

    def forward(self, input, hidden, bias=0):
        # batch size * sequence len * features (3)
        output, hidden = self.lstm(input, hidden)
        output = self.linear(output)
        output = self.bgm(output, bias=bias)
        return output + (hidden,)

    def init_hidden(self, batch_size=1):
        if self.is_cuda():
            hidden = Variable(torch.zeros(1, batch_size, 900).cuda())
            cell = Variable(torch.zeros(1, batch_size, 900).cuda())
        else:
            hidden = Variable(torch.zeros(1, batch_size, 900))
            cell = Variable(torch.zeros(1, batch_size, 900))
        return [hidden, cell]

    def is_cuda(self):
        return next(self.parameters()).is_cuda

class WindowLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, num_components=1, num_chars=26):
        super(WindowLayer,self).__init__()
        self.num_chars = num_chars
        self.num_components = num_components
        self.linear = torch.nn.Linear(in_features=in_features,out_features=3*self.num_components)

    def forward(self, inputs, hidden, sequence):
        """
        sequence -- An array of one-hot encodings for characters in 
        """
        n = self.num_components
        output = self.linear(inputs)
        output = torch.exp(output)
        output = output.view(-1,3,n)
        hidden = hidden + output[:,2,:]
        a = output[:,0,:].contiguous().view(-1,1,n)
        b = output[:,1,:].contiguous().view(-1,1,n)
        k = hidden.contiguous().view(-1,1,n)
        seq_len = sequence.size()[0]
        if self.is_cuda():
            r = Variable(torch.arange(seq_len).view(seq_len,1).cuda(), requires_grad=False)
        else:
            r = Variable(torch.arange(seq_len).view(seq_len,1), requires_grad=False)
        window_weight = a*torch.exp(-b*torch.pow(k-r,2))
        window_weight = torch.sum(window_weight,dim=2)
        window = window_weight.mm(sequence) # [-1,seq_len] * [seq_len, features]

        terminal = WindowLayer.is_terminal(k, seq_len)

        return window, hidden, terminal

    def init_hidden(self):
        if self.is_cuda():
            return Variable(torch.zeros(1,self.num_components).cuda())
        else:
            return Variable(torch.zeros(1,self.num_components))

    @staticmethod
    def is_terminal(k, seq_len):
        return (k>seq_len-0.5).all()

    def is_cuda(self):
        return next(self.parameters()).is_cuda

class ConditionedRNN(torch.nn.Module):
    def __init__(self, num_components=1, num_chars=27):
        super(ConditionedRNN,self).__init__()
        # See page 23 for parameters
        self.num_chars = num_chars
        self.num_components = num_components
        self.hidden_size = 400
        
        self.lstm1 = torch.nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                num_layers=1, bias=True)
        self.window = WindowLayer(in_features=self.hidden_size,out_features=self.hidden_size,
                num_components=10, num_chars=num_chars)
        self.lstm2 = torch.nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                num_layers=1, bias=True)

        self.linearx1 = torch.nn.Linear(in_features=3,out_features=self.hidden_size,
                bias=False)
        self.linearx2 = torch.nn.Linear(in_features=3,out_features=self.hidden_size,
                bias=False)
        self.linearw1 = torch.nn.Linear(in_features=num_chars,out_features=self.hidden_size,
                bias=False)
        self.linearw2 = torch.nn.Linear(in_features=num_chars,out_features=self.hidden_size,
                bias=False)
        self.linear12 = torch.nn.Linear(in_features=self.hidden_size,out_features=self.hidden_size,
                bias=False)
        self.linearout = torch.nn.Linear(in_features=self.hidden_size,
                out_features=1+(1+2+2+1)*self.num_components)
        self.bgm = BivariateGaussianMixtureLayer(self.num_components)

    def forward(self, input, hidden, sequence, bias=0):
        hidden1,hiddenw,outputw,hidden2 = hidden

        input1 = self.linearx1(input)+self.linearw1(outputw)
        output1, hidden1 = self.lstm1(input1, hidden1)

        outputw, hiddenw, terminal = self.window(output1, hiddenw, sequence)

        input2 = self.linearx2(input)+self.linear12(output1)+self.linearw2(outputw)
        output2, hidden2 = self.lstm2(input2, hidden2)

        inputbgm = self.linearout(output2)
        outputbgm = self.bgm(inputbgm, bias=bias)
        return outputbgm + ((hidden1,hiddenw,outputw,hidden2),) + (terminal,)

    def init_hidden(self):
        if self.is_cuda():
            hidden1 = Variable(torch.zeros(1, 1, 400).cuda())
            cell1 = Variable(torch.zeros(1, 1, 400).cuda())
            outputw = Variable(torch.zeros(1, self.num_chars).cuda())
            hidden2 = Variable(torch.zeros(1, 1, 400).cuda())
            cell2 = Variable(torch.zeros(1, 1, 400).cuda())
        else:
            hidden1 = Variable(torch.zeros(1, 1, 400))
            cell1 = Variable(torch.zeros(1, 1, 400))
            outputw = Variable(torch.zeros(1, self.num_chars))
            hidden2 = Variable(torch.zeros(1, 1, 400))
            cell2 = Variable(torch.zeros(1, 1, 400))
        hiddenw = self.window.init_hidden()
        return (hidden1, cell1), hiddenw, outputw, (hidden2, cell2)

    def is_cuda(self):
        return next(self.parameters()).is_cuda

def generate_sequence(rnn : GeneratorRNN,
        length : int,
        start=[0,0,0],
        bias: int=0):
    """
    Generate a random sequence of handwritten strokes, with `length` strokes.
    """
    if rnn.is_cuda():
        inputs = Variable(torch.Tensor(start).view(1,1,3).float().cuda())
    else:
        inputs = Variable(torch.Tensor(start).view(1,1,3).float())
    hidden = rnn.init_hidden()
    strokes = np.empty([length+1,3])
    strokes[0] = start
    for i in range(1,length+1):
        e,pi,mu,sigma,rho,hidden = rnn.forward(inputs, hidden, bias=bias)

        # Sample from bivariate Gaussian mixture model
        # Choose a component
        pi = pi[0,0,:].view(-1).data.cpu().numpy()
        component = np.random.choice(range(rnn.num_components), p=pi)
        if rnn.is_cuda():
            mu = mu[0,0,component].data.cpu().numpy()
            sigma = sigma[0,0,component].data.cpu().numpy()
            rho = rho[0,0,component].data.cpu().numpy()
        else:
            mu = mu[0,0,component].data.numpy()
            sigma = sigma[0,0,component].data.numpy()
            rho = rho[0,0,component].data.numpy()

        # Sample from the selected Gaussian
        covar = [[sigma[0]**2, rho[0]*sigma[0]*sigma[1]],
                [rho[0]*sigma[0]*sigma[1], sigma[1]**2]] # See https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        sample = np.random.multivariate_normal(mu,covar)

        # Sample from Bernoulli
        if rnn.is_cuda():
            e = e.data.cpu().numpy()
        else:
            e = e.data.numpy()
        lift = np.random.binomial(1,e)[0]

        # Store stroke
        strokes[i] = [lift,sample[0],sample[1]]

        # Update next input
        inputs.data[0][0][0] = int(lift)
        inputs.data[0][0][1] = sample[0]
        inputs.data[0][0][2] = sample[1]

    return strokes

def generate_conditioned_sequence(rnn : ConditionedRNN, length : int,
        sentence, start = [0,0,0], bias: int = 0):
    """
    Generate a sequence of handwritten strokes representing the given sentence, with at most `length` strokes.
    """
    if rnn.is_cuda():
        inputs = Variable(torch.Tensor(start).view(1,1,3).float().cuda())
    else:
        inputs = Variable(torch.Tensor(start).view(1,1,3).float())
    hidden = rnn.init_hidden()
    strokes = np.empty([length+1,3])
    strokes[0] = start
    terminal = False
    for i in range(1,length+1):
        e,pi,mu,sigma,rho,hidden,terminal = rnn.forward(inputs, hidden, sentence, bias=bias)
        if terminal:
            strokes = strokes[:i,:]
            break

        # Sample from bivariate Gaussian mixture model
        # Choose a component
        pi = pi[0,0,:].view(-1).data.cpu().numpy()
        component = np.random.choice(range(rnn.num_components), p=pi)
        if rnn.is_cuda():
            mu = mu[0,0,component].data.cpu().numpy()
            sigma = sigma0,[0,component].data.cpu().numpy()
            rho = rho[0,0,component].data.cpu().numpy()
        else:
            mu = mu[0,0,component].data.numpy()
            sigma = sigma[0,0,component].data.numpy()
            rho = rho[0,0,component].data.numpy()

        # Sample from the selected Gaussian
        covar = [[sigma[0]**2, rho*sigma[0]*sigma[1]],
                [rho*sigma[0]*sigma[1], sigma[1]**2]] # See https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        sample = np.random.multivariate_normal(mu,covar)

        # Sample from Bernoulli
        if rnn.is_cuda():
            e = e.data.cpu().numpy()
        else:
            e = e.data.numpy()
        lift = np.random.binomial(1,e)[0]

        # Store stroke
        strokes[i] = [lift,sample[0],sample[1]]

        # Update next input
        inputs.data[0][0][0] = int(lift)
        inputs.data[0][0][1] = sample[0]
        inputs.data[0][0][2] = sample[1]

    return strokes

def prob(x, y):
    """
    Return the probability of the next point being x given that parameters y
    were outputted by the neural net.
    See equation (23)
    """
    e,pi,mu,sigma,rho = y
    num_components = mu.size()[2]
    p = 0
    for i in range(num_components):
        p += pi[:,:,i]*normal(x[:,:,1:], mu[:,:,i,:],sigma[:,:,i,:],rho[:,:,i])
    p *= (x[0,:,0]==1).float()*e+(x[0,:,0]==0).float()*(1-e)
    #p *= torch.where(x[0,:,0]==1,e,1-e)
    #if (p<=0.00000001).any():
    #    if p.is_cuda:
    #        p=Variable(torch.ones(p.size()).cuda(), requires_grad=False)
    #    else:
    #        p=Variable(torch.ones(p.size()), requires_grad=False)
    return p

def normal(x, mu, sigma, rho):
    z  = torch.pow((x[:,:,0]-mu[:,:,0])/sigma[:,:,0],2)
    z += torch.pow((x[:,:,1]-mu[:,:,1])/sigma[:,:,1],2)
    #z = torch.pow((x-mu)/sigma,2)
    z -= 2*rho*(x[:,:,0]-mu[:,:,0])*(x[:,:,1]-mu[:,:,1])/(sigma[:,:,0]*sigma[:,:,1])
    output = 1/(2*np.pi*sigma[:,:,0]*sigma[:,:,1]*torch.sqrt(1-rho*rho))*torch.exp(-z/(2*(1-rho*rho)))
    return output

def print_avg_grad(rnn: GeneratorRNN):
    vals = []
    for p in rnn.parameters():
        vals += list(np.abs(p.grad.view(-1).data.cpu().numpy()))
    print('Average grad: %f' % np.mean(vals))

def batch(strokes):
    batch_size = len(strokes)
    max_len = max([len(s) for s in strokes])
    batched_strokes = np.empty([max_len, batch_size,3])
    for i,s in enumerate(strokes):
        batched_strokes[:len(s),i,:] = s
    mask = np.array([[1]*len(s)+[0]*(max_len-len(s)) for s in strokes]).transpose()
    return batched_strokes, mask

def batch_min(strokes):
    batch_size = len(strokes)
    max_len = min([len(s) for s in strokes])
    batched_strokes = np.empty([max_len, batch_size,3])
    for i,s in enumerate(strokes):
        batched_strokes[:,i,:] = s[:max_len,:]
    mask = np.array([[1]*max_len for s in strokes]).transpose()
    return batched_strokes, mask

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
    for i in tqdm(range(len(strokes)-1)):
        x = strokes_var[i].view(1,-1,3)
        e,pi,mu,sigma,rho,hidden = rnn(x, hidden)
        y = (e,pi,mu,sigma,rho)
        x2 = strokes_var[i+1].view(1,-1,3)

        likelihood = torch.masked_select(prob(x2,y),mask[i,:])
        likelihood = likelihood[(likelihood > 0.00000001).detach()]
        loss = -torch.log(likelihood)
        total_loss += torch.sum(loss)
        #loss = -torch.log(prob(x2,y))
        #print(loss.size())
        ##total_loss += loss
        #for b in range(batch_size):
        #    if (mask[i,b] == 1).all():
        #        total_loss[i] += loss[0,i]

    return total_loss

def compute_loss_conditioned(rnn, strokes, sentence):
    strokes_tensor = torch.from_numpy(strokes).float()
    if rnn.is_cuda():
        strokes_tensor = strokes_tensor.cuda()
    strokes_var = Variable(strokes_tensor, requires_grad=False)
    hidden = rnn.init_hidden()
    total_loss = 0
    for i in tqdm(range(len(strokes)-1)):
        x = strokes_var[i].view(1,1,3)
        e,pi,mu,sigma,rho,hidden,_ = rnn(x, hidden, sentence)
        y = (e,pi,mu,sigma,rho)
        x2 = strokes_var[i+1].view(1,1,3)

        loss = -torch.log(prob(x2,y))
        total_loss += loss

    return total_loss

def train(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, strokes):
    total_loss = compute_loss(rnn, strokes)

    optimizer.zero_grad()
    tqdm.write("Loss: %s" % (total_loss.data[0]/len(strokes)))
    total_loss.backward()
    torch.nn.utils.clip_grad.clip_grad_norm(rnn.parameters(), 10)
    optimizer.step()

def train_batch(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, strokes):
    batched_strokes, mask = batch(strokes)
    if rnn.is_cuda():
        mask = Variable(torch.from_numpy(mask).byte().cuda(), requires_grad=False)
    else:
        mask = Variable(torch.from_numpy(mask).byte(), requires_grad=False)
    total_loss = compute_loss(rnn, batched_strokes, mask)
    total_loss = compute_loss(rnn, batched_strokes)
    #total_loss = torch.sum(total_loss)

    optimizer.zero_grad()
    tqdm.write("Loss: %s" % (total_loss.data[0]/len(strokes)))
    total_loss.backward()
    torch.nn.utils.clip_grad.clip_grad_norm(rnn.parameters(), 10)
    optimizer.step()

def train_all(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, data, sm, ss):
    i = 0
    while True:
        for strokes in tqdm(data):
            if i%50 == 0:
                for b in [0,0.1,10]:
                    generated_strokes = generate_sequence(rnn, 700,
                            [0,sm[1],sm[2]], bias=b)
                    generated_strokes = unnormalize_strokes(generated_strokes, sm, ss)
                    file_name = 'output/%d-%s.png'%(i,b)
                    plot_stroke(generated_strokes, file_name)
                    tqdm.write('Writing file: %s' % file_name)
            i+=1
            train(rnn, optimizer, strokes)
    return

def train_all_random_batch(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, data, sm, ss):
    batch_size=50
    i = 0
    pbar = tqdm()
    while True:
        batched_data = np.random.choice(data,size=batch_size,replace=False)
        if i%50 == 0:
            for b in [0,0.1,5,10]:
                generated_strokes = generate_sequence(rnn, 500,
                        [0,sm[1],sm[2]], bias=b)
                generated_strokes = unnormalize_strokes(generated_strokes, sm, ss)
                file_name = 'output_batch2/%d-%s.png'%(i,b)
                plot_stroke(generated_strokes, file_name)
                tqdm.write('Writing file: %s' % file_name)
        i+=1
        train_batch(rnn, optimizer, batched_data)
        pbar.update(1)
    return

def train_one(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, strokes, sm, ss):
    i = 0
    pbar = tqdm()
    pbar.update(0)
    while True:
        if i%50 == 0:
                for b in [0,0.1,10]:
                    generated_strokes = generate_sequence(rnn, 700,
                            [0,sm[1],sm[2]], bias=b)
                    generated_strokes = unnormalize_strokes(generated_strokes, sm, ss)
                    file_name = 'output/%d-%s.png'%(i,b)
                    plot_stroke(generated_strokes, file_name)
                    tqdm.write('Writing file: %s' % file_name)
        i+=1
        train(rnn, optimizer, strokes)
        pbar.update(1)
    return

def train_conditioned(rnn : ConditionedRNN, optimizer : torch.optim.Optimizer,
        strokes, sentence):
    total_loss = compute_loss_conditioned(rnn, strokes, sentence)

    optimizer.zero_grad()
    tqdm.write("Loss: %s" % (total_loss.data[0]/len(strokes)))
    total_loss.backward()
    torch.nn.utils.clip_grad.clip_grad_norm(rnn.parameters(), 10)
    optimizer.step()

def train_all_conditioned(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, data, sentences, sm, ss, target_sentence=None):
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
            train_conditioned(rnn, optimizer, strokes, sentence)
    return

def create_optimizer(rnn):
    # Paper says they're using RMSProp, but the equations (38)-(41) look like Adam with momentum.
    # See parameters in equation (42)-(45)
    # Reference https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
    # Reference http://pytorch.org/docs/master/optim.html
    # (42) is Wikipedia's beta1 and beta2
    # (43) is momentum
    # (44) is learning rate
    # (45) is epsilon (added to denom for numerical stability)
    # Skipped out on Momentum, since it's not implemented by pytorch
    optimizer = torch.optim.Adam(params=rnn.parameters(),lr=0.0001,betas=(0.95,0.95),eps=0.0001)
    #optimizer = torch.optim.RMSprop(params=rnn.parameters(),lr=0.0001,alpha=0.95,eps=0.0001)
    #optimizer = torch.optim.SGD(params=rnn.parameters(),lr=0.0001)
    return optimizer

if __name__=='__main__':
    data = load_data()
    normalized_data,m,s = normalize_strokes(data)
    sentences = load_sentences()
    alphabet, alphabet_dict = compute_alphabet(sentences)
    sentence_vars = [Variable(torch.from_numpy(sentence_to_vectors(s,alphabet_dict)).float().cuda(),
        requires_grad=False) for s in tqdm(sentences,desc="Converting Sentences")]
    rnn = GeneratorRNN(20).cuda()
    #rnn = ConditionedRNN(20, len(alphabet)).cuda()
    #rnn.load_state_dict(torch.load('model-all.pt'))
    #rnn.load_state_dict(torch.load('models_cond/4850.pt'))

    optimizer = create_optimizer(rnn)

    #train_all(rnn, optimizer, normalized_data, m.tolist(), s.tolist())
    #train_one(rnn, optimizer, normalized_data[0], m.tolist(), s.tolist())
    #train(rnn, optimizer, normalized_data[0])
    train_all_random_batch(rnn, optimizer, normalized_data, m.tolist(), s.tolist())

    #target_sentence = Variable(torch.from_numpy(sentence_to_vectors("Hello World!",alphabet_dict)).float().cuda(), requires_grad=False)
    #train_all_conditioned(rnn, optimizer, normalized_data, sentence_vars, m.tolist(),
    #        s.tolist(), target_sentence=target_sentence)
    #train_conditioned(rnn, optimizer, normalized_data[0], sentence_vars[0])
