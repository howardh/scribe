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
    print(concatenated.shape)
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

    def forward(self, input):
        # batch size * sequence len * features (3)
        e,pi,mu,sigma,rho = self.split_outputs(input)

        n = self.num_components
        e = 1/(1+torch.exp(e)).view(-1)
        pi = self.softmax(pi).view(-1,n)
        #pi = Variable(torch.zeros([n]).view(-1,n).cuda())
        mu = mu.contiguous().view(-1,n,2)
        sigma = torch.exp(sigma).view(-1,n,2)
        rho = torch.tanh(rho).view(-1,n)
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

    def forward(self, input, hidden):
        # batch size * sequence len * features (3)
        output, hidden = self.lstm(input, hidden)
        output = self.linear(output)
        output = self.bgm(output)
        print(len(output + (hidden)))
        return output + (hidden,)

    def init_hidden(self):
        if self.is_cuda():
            hidden = Variable(torch.zeros(1, 1, 900).cuda())
            cell = Variable(torch.zeros(1, 1, 900).cuda())
        else:
            hidden = Variable(torch.zeros(1, 1, 900))
            cell = Variable(torch.zeros(1, 1, 900))
        return [hidden, cell]

    def is_cuda(self):
        return next(self.parameters()).is_cuda

class WindowLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, num_components=1, num_chars=26):
        super(WindowLayer,self).__init__()
        self.num_chars = num_chars
        self.num_components = num_components
        self.linear1 = torch.nn.Linear(in_features=in_features,out_features=3*self.num_components)
        self.linear2 = torch.nn.Linear(in_features=self.num_chars,out_features=out_features)

    def forward(self, inputs, hidden, sequence):
        """
        sequence -- An array of one-hot encodings for characters in 
        """
        n = self.num_components
        output = self.linear1(inputs)
        output = torch.exp(output)
        output = output.view(-1,3,n)
        hidden = hidden + output[:,2,:]
        output[:,2,:] = hidden
        a = output[:,0,:].contiguous().view(-1,1,n)
        b = output[:,1,:].contiguous().view(-1,1,n)
        k = output[:,2,:].contiguous().view(-1,1,n)
        seq_len = sequence.size()[0]
        if self.is_cuda():
            r = Variable(torch.arange(seq_len).view(seq_len,1).cuda(), requires_grad=False)
        else:
            r = Variable(torch.arange(seq_len).view(seq_len,1), requires_grad=False)
        window_weight = a*torch.exp(-b*torch.pow(k-r,2))
        window_weight = torch.sum(window_weight,dim=2)
        window = window_weight.mm(sequence)
        output = self.linear2(window)
        return output, hidden

    def init_hidden(self):
        if self.is_cuda():
            return Variable(torch.zeros(1,1,self.num_components).cuda())
        else:
            return Variable(torch.zeros(1,1,self.num_components))

    def is_cuda(self):
        return next(self.parameters()).is_cuda

class ConditionedRNN(torch.nn.Module):
    def __init__(self, num_components=1, num_chars=27):
        super(ConditionedRNN,self).__init__()
        # See page 23 for parameters
        self.num_components = num_components
        self.lstm1 = torch.nn.LSTM(input_size=3, hidden_size=400, num_layers=1, bias=True)
        self.window = WindowLayer(in_features=400,out_features=400,
                num_components=10, num_chars=num_chars)
        self.lstm2 = torch.nn.LSTM(input_size=400, hidden_size=400, num_layers=1, bias=True)
        self.linearx2 = torch.nn.Linear(in_features=3,out_features=400,
                bias=False)
        self.linear12 = torch.nn.Linear(in_features=400,out_features=400,
                bias=False)
        self.linearout = torch.nn.Linear(in_features=400,out_features=1+(1+2+2+1)*self.num_components)
        self.bgm = BivariateGaussianMixtureLayer(num_components)

        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, input, hidden, sequence):
        hidden1,hiddenw,hidden2 = hidden
        output1, hidden1 = self.lstm1(input, hidden1)
        outputw, hiddenw = self.window(output1, hiddenw, sequence)
        input2 = self.linearx2(input)+output1
        output2, hidden2 = self.lstm2(input2, hidden2)
        outputbgm = self.bgm(output2)
        return outputbgm + ((hidden1,hiddenw,hidden2),)

    def init_hidden(self):
        if self.is_cuda():
            hidden1 = Variable(torch.zeros(1, 1, 400).cuda())
            cell1 = Variable(torch.zeros(1, 1, 400).cuda())
            hidden2 = Variable(torch.zeros(1, 1, 400).cuda())
            cell2 = Variable(torch.zeros(1, 1, 400).cuda())
        else:
            hidden1 = Variable(torch.zeros(1, 1, 400))
            cell1 = Variable(torch.zeros(1, 1, 400))
            hidden2 = Variable(torch.zeros(1, 1, 400))
            cell2 = Variable(torch.zeros(1, 1, 400))
        hiddenw = self.window.init_hidden()
        return (hidden1, cell1), hiddenw, (hidden2, cell2)

    def is_cuda(self):
        return next(self.parameters()).is_cuda

def generate_sequence(rnn : GeneratorRNN, length : int, start = [0,0,0]):
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
        e,pi,mu,sigma,rho,hidden = rnn.forward(inputs, hidden)

        # Sample from bivariate Gaussian mixture model
        # Choose a component
        pi = pi[0,:].view(-1).data.cpu().numpy()
        component = np.random.choice(range(rnn.num_components), p=pi)
        if rnn.is_cuda():
            mu = mu[0,component].data.cpu().numpy()
            sigma = sigma[0,component].data.cpu().numpy()
            rho = rho[0,component].data.cpu().numpy()
        else:
            mu = mu[0,component].data.numpy()
            sigma = sigma[0,component].data.numpy()
            rho = rho[0,component].data.numpy()

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

def generate_conditioned_sequence(rnn : ConditionedRNN, length : int,
        sentence, start = [0,0,0]):
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
        e,pi,mu,sigma,rho,hidden = rnn.forward(inputs, hidden, sentence)

        # Sample from bivariate Gaussian mixture model
        # Choose a component
        pi = pi[0,:].view(-1).data.cpu().numpy()
        component = np.random.choice(range(rnn.num_components), p=pi)
        if rnn.is_cuda():
            mu = mu[0,component].data.cpu().numpy()
            sigma = sigma[0,component].data.cpu().numpy()
            rho = rho[0,component].data.cpu().numpy()
        else:
            mu = mu[0,component].data.numpy()
            sigma = sigma[0,component].data.numpy()
            rho = rho[0,component].data.numpy()

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
    num_components = mu.size()[1]
    p = 0
    for i in range(num_components):
        p += pi[:,i]*normal(x[:,0,1:], mu[:,i,:],sigma[:,i,:],rho[:,i])

    if x.is_cuda:
        temp = x[0,0,0].data.cpu().numpy()[0]
    else:
        temp = x[0,0,0].data.numpy()[0]
    if temp == 1:
        p *= e
    else:
        p *= (1-e)
    return p

def normal(x, mu, sigma, rho):
    z  = torch.pow((x[:,0]-mu[:,0])/sigma[:,0],2)
    z += torch.pow((x[:,1]-mu[:,1])/sigma[:,1],2)
    #z = torch.pow((x-mu)/sigma,2)
    z -= 2*rho*(x[:,0]-mu[:,0])*(x[:,1]-mu[:,1])/(sigma[:,0]*sigma[:,1])
    output = 1/(2*np.pi*sigma[:,0]*sigma[:,1]*torch.sqrt(1-rho*rho))*torch.exp(-z/(2*(1-rho*rho)))
    return output

def print_avg_grad(rnn: GeneratorRNN):
    vals = []
    for p in rnn.parameters():
        vals += list(np.abs(p.grad.view(-1).data.cpu().numpy()))
    print('Average grad: %f' % np.mean(vals))

def compute_loss(rnn, strokes):
    strokes_tensor = torch.from_numpy(strokes).float()
    if rnn.is_cuda():
        strokes_tensor = strokes_tensor.cuda()
    strokes_var = Variable(strokes_tensor, requires_grad=False)
    hidden = rnn.init_hidden()
    total_loss = 0
    eps = 0.00001
    for i in tqdm(range(len(strokes)-1)):
        x = strokes_var[i].view(1,1,3)
        e,pi,mu,sigma,rho,hidden = rnn(x, hidden)
        y = (e,pi,mu,sigma,rho)
        x2 = strokes_var[i+1].view(1,1,3)

        loss = -torch.log(prob(x2,y)+eps)
        total_loss += loss

    return total_loss

def compute_loss_conditioned(rnn, strokes, sentence):
    strokes_tensor = torch.from_numpy(strokes).float()
    if rnn.is_cuda():
        strokes_tensor = strokes_tensor.cuda()
    strokes_var = Variable(strokes_tensor, requires_grad=False)
    hidden = rnn.init_hidden()
    total_loss = 0
    eps = 0.00001
    for i in tqdm(range(len(strokes)-1)):
        x = strokes_var[i].view(1,1,3)
        e,pi,mu,sigma,rho,hidden = rnn(x, hidden, sentence)
        y = (e,pi,mu,sigma,rho)
        x2 = strokes_var[i+1].view(1,1,3)

        loss = -torch.log(prob(x2,y)+eps)
        total_loss += loss

    return total_loss

def train(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, strokes):
    total_loss = compute_loss(rnn, strokes)

    optimizer.zero_grad()
    tqdm.write("Loss: %s" % total_loss.data[0])
    total_loss.backward()
    torch.nn.utils.clip_grad.clip_grad_norm(rnn.parameters(), 10)
    optimizer.step()

def train_all(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, data, sm, ss):
    i = 0
    while True:
        for strokes in tqdm(data):
            if i%50 == 0:
                generated_strokes = generate_sequence(rnn, 700, [0,sm[1],sm[2]])
                generated_strokes = unnormalize_strokes(generated_strokes, sm, ss)
                plot_stroke(generated_strokes, 'output/%d.png'%i)
            i+=1
            train(rnn, optimizer, strokes)
    return

def train_one(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, strokes, unnorm):
    i = 0
    pbar = tqdm()
    pbar.update(0)
    while True:
        if i%50 == 0:
            generated_strokes = generate_sequence(rnn, 700)
            generated_strokes = unnorm(generated_strokes)
            plot_stroke(generated_strokes, 'output/%d.png'%i)
        i+=1
        train(rnn, optimizer, strokes)
        pbar.update(1)
    return

def train_conditioned(rnn : ConditionedRNN, optimizer : torch.optim.Optimizer,
        strokes, sentence):
    total_loss = compute_loss_conditioned(rnn, strokes, sentence)

    optimizer.zero_grad()
    tqdm.write("Loss: %s" % total_loss.data[0])
    total_loss.backward()
    torch.nn.utils.clip_grad.clip_grad_norm(rnn.parameters(), 10)
    optimizer.step()

def train_all_conditioned(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, data, sentences, sm, ss, target_sentence=None):
    i = 0
    while True:
        for strokes, sentence in tqdm(zip(data, sentences)):
            if target_sentence is not None and i%50 == 0:
                generated_strokes = generate_conditioned_sequence(rnn, 700,
                        target_sentence, [0,sm[1],sm[2]])
                generated_strokes = unnormalize_strokes(generated_strokes, sm, ss)
                plot_stroke(generated_strokes, 'output_cond/%d.png'%i)
                torch.save(rnn.state_dict(), "models_cond/%d.pt"%i)
            i+=1
            train_conditioned(rnn, optimizer, strokes, sentence)
    return

def foo():
    # Check what plot thing does
    x = np.array([[0,0,0],[0,1,1],[0,1,0],[0,1,1],[1,0,0]])
    plot_stroke(x,'test.png')

if __name__=='__main__':
    data = load_data()
    normalized_data,m,s = normalize_strokes(data)
    sentences = load_sentences()
    alphabet, alphabet_dict = compute_alphabet(sentences)
    sentence_vars = [Variable(torch.from_numpy(sentence_to_vectors(s,alphabet_dict)).float().cuda(),
        requires_grad=False) for s in tqdm(sentences,desc="Converting Sentences")]
    #rnn = GeneratorRNN(20)
    rnn = ConditionedRNN(20, len(alphabet))
    rnn.cuda()
    #rnn.load_state_dict(torch.load('model.pt'))

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

    #train_all(rnn, optimizer, normalized_data, m.tolist(), s.tolist())
    #train_one(rnn, optimizer, normalized_data[0], lambda strokes: unnormalize_strokes(strokes,m,s))
    #train(rnn, optimizer, normalized_data[0])

    target_sentence = Variable(torch.from_numpy(sentence_to_vectors("Hello World!",alphabet_dict)).float().cuda(), requires_grad=False)
    train_all_conditioned(rnn, optimizer, normalized_data, sentence_vars, m.tolist(),
            s.tolist(), target_sentence=target_sentence)
    #train_conditioned(rnn, optimizer, normalized_data[0], sentence_vars[0])
