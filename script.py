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

def normalize_strokes(strokes):
    m = np.mean(strokes, axis=0)
    s = np.std(strokes, axis=0)
    output = np.empty(strokes.shape)
    for i in range(len(strokes)):
        output[i][0] = strokes[i][0]
        output[i][1] = (strokes[i][1]-m[1])/s[1]
        output[i][2] = (strokes[i][2]-m[2])/s[2]
    return output, m, s

def unnormalize_strokes(strokes, m, s):
    output = np.empty(strokes.shape)
    for i in range(len(strokes)):
        output[i][0] = strokes[i][0]
        output[i][1] = strokes[i][1]*s[1]+m[1]
        output[i][2] = strokes[i][2]*s[2]+m[2]
    return output

class GeneratorRNN(torch.nn.Module):
    def __init__(self, num_components=1, use_cuda=False):
        super(GeneratorRNN,self).__init__()
        self.use_cuda = use_cuda
        # See page 23 for parameters
        self.num_components = num_components
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=900, num_layers=1,
                bias=True)
        self.linear = torch.nn.Linear(in_features=900,out_features=1+(1+2+2+1)*self.num_components)
        self.softmax = torch.nn.Softmax(dim=2)

        if self.use_cuda:
            self.lstm = self.lstm.cuda()
            self.linear = self.linear.cuda()
            self.softmax = self.softmax.cuda()

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

    def forward(self, input, hidden):
        # batch size * sequence len * features (3)
        output, hidden = self.lstm(input, hidden)
        output = self.linear(output)
        e,pi,mu,sigma,rho = self.split_outputs(output)

        n = self.num_components
        e = 1/(1+torch.exp(e)).view(-1)
        pi = self.softmax(pi).view(-1,n)
        #pi = Variable(torch.zeros([n]).view(-1,n).cuda())
        mu = mu.contiguous().view(-1,n,2)
        sigma = torch.exp(sigma).view(-1,n,2)
        rho = torch.tanh(rho).view(-1,n)
        return e,pi,mu,sigma,rho,hidden

    def init_hidden(self):
        if self.use_cuda:
            hidden = Variable(torch.zeros(1, 1, 900).cuda())
            cell = Variable(torch.zeros(1, 1, 900).cuda())
        else:
            hidden = Variable(torch.zeros(1, 1, 900))
            cell = Variable(torch.zeros(1, 1, 900))
        return [hidden, cell]

def generate_sequence(rnn : GeneratorRNN, length : int):
    """
    Generate a random sequence of handwritten strokes, with `length` strokes.
    """
    if rnn.use_cuda:
        inputs = Variable(torch.zeros(1,1,3).float().cuda())
    else:
        inputs = Variable(torch.zeros(1,1,3).float())
    hidden = rnn.init_hidden()
    strokes = np.empty([length+1,3])
    strokes[0] = [0,0,0]
    for i in range(1,length+1):
        e,pi,mu,sigma,rho,hidden = rnn.forward(inputs, hidden)

        # Sample from bivariate Gaussian mixture model
        # Choose a component
        pi = pi[0,:].view(-1).data.cpu().numpy()
        component = np.random.choice(range(rnn.num_components), p=pi)
        if rnn.use_cuda:
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
        if rnn.use_cuda:
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

def relative_to_absolute(strokes):
    output = np.copy(strokes)
    for i in range(1,len(strokes)):
        output[i] = [strokes[i][0],
                strokes[i][1]+strokes[i-1][1],
                strokes[i][2]+strokes[i-1][2]]
    return output

def absolute_to_relative(strokes):
    output = np.copy(strokes)
    for i in reversed(range(1,len(strokes))):
        output[i] = [strokes[i][0],
                strokes[i][1]-strokes[i-1][1],
                strokes[i][2]-strokes[i-1][2]]
    return output

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
    if rnn.use_cuda:
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
        #loss = -torch.log(prob(x2,y))
        #loss = -prob(x2,y)
        total_loss += loss

    return total_loss

def train(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, strokes):
    total_loss = compute_loss(rnn, strokes)

    optimizer.zero_grad()
    print("Loss: ", total_loss)
    total_loss.backward()
    torch.nn.utils.clip_grad.clip_grad_norm(rnn.parameters(), 10)
    #print(rnn.parameters().__next__()[0][0])
    optimizer.step()

def train_all(rnn : GeneratorRNN, optimizer : torch.optim.Optimizer, data, unnorm):
    i = 0
    for strokes in tqdm(data):
        generated_strokes = generate_sequence(rnn, 100)
        generated_strokes = unnorm(generated_strokes)
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
            generated_strokes = generate_sequence(rnn, 750)
            generated_strokes = unnorm(generated_strokes)
            plot_stroke(generated_strokes, 'output/%d.png'%i)
        i+=1
        train(rnn, optimizer, strokes)
        pbar.update(1)
    return

def foo():
    # Check what plot thing does
    x = np.array([[0,0,0],[0,1,1],[0,1,0],[0,1,1],[1,0,0]])
    plot_stroke(x,'test.png')

if __name__=='__main__':
    data = load_data()
    x,m,s = normalize_strokes(data[0])
    l = [len(d) for d in data]
    print(max(l)) #1191
    rnn = GeneratorRNN(20, use_cuda=True)
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

    #train_all(rnn, data)
    train_one(rnn, optimizer, x, lambda strokes: unnormalize_strokes(strokes,m,s))
    #train(rnn, optimizer, x)
