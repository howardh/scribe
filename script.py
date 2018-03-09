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

class GeneratorRNN(torch.nn.Module):
    def __init__(self, num_components=1):
        super(GeneratorRNN,self).__init__()
        # See page 23 for parameters
        self.num_components = num_components
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=900, num_layers=1,
                bias=True)
        self.linear = torch.nn.Linear(in_features=900,out_features=1+(1+2+2+1)*self.num_components)
        self.softmax = torch.nn.Softmax(dim=0)

    def split_outputs(self, output):
        e = output[0]

        start = 1
        end = 1+self.num_components
        pi = output[start:end]

        start = end
        end = end+2*self.num_components
        mu = output[start:end].view(-1,2)

        start = end
        end = end+2*self.num_components
        sigma = output[start:end].view(-1,2)

        start = end
        end = end+self.num_components
        rho = output[start:end]

        return e,pi,mu,sigma,rho

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.linear(output)
        output = output.view(-1)
        e,pi,mu,sigma,rho = self.split_outputs(output)
        e = 1/(1+torch.exp(e))
        pi = self.softmax(pi)
        mu = mu
        sigma = torch.exp(sigma)
        rho = torch.tanh(rho)
        return e,pi,mu,sigma,rho,hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(1, 1, 900))
        cell = Variable(torch.zeros(1, 1, 900))
        return [hidden, cell]

def generate_sequence(rnn : GeneratorRNN, length : int):
    """
    Generate a random sequence of handwritten strokes, with `length` strokes.
    """
    inputs = Variable(torch.zeros(1,3).float())
    hidden = rnn.init_hidden()
    strokes = np.empty([length+1,3])
    strokes[0] = [0,0,0]
    for i in range(1,length+1):
        e,pi,mu,sigma,rho,hidden = rnn.forward(inputs, hidden)

        e = e.data.numpy()
        rho = rho.data.numpy()

        # Sample from bivariate Gaussian mixture model
        # Choose a component
        component = np.random.choice(range(rnn.num_components))
        mu = mu[component].data.numpy()
        sigma = sigma[component].data.numpy()

        # Sample from the selected Gaussian
        covar = [[sigma[0]**2, rho*sigma[0]*sigma[1]],
                [rho*sigma[0]*sigma[1], sigma[1]**2]] # See https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        sample = np.random.multivariate_normal(mu,covar)

        # Sample from Bernoulli
        lift = np.random.binomial(1,e)

        # Store stroke
        strokes[i] = [lift,sample[0],sample[1]]

    return relative_to_absolute(strokes)

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
    num_components = mu.size()[0]
    p = 1
    for i in range(num_components):
        p *= pi[i]*normal(x, mu[i],sigma[i],rho)
        if x[0][0].data.numpy()[0] == 1:
            p *= e
        else:
            p *= (1-e)
    return p

def normal(x, mu, sigma, rho):
    x = x[0][1:]
    z = torch.pow((x[0]-mu[0])/sigma[0],2)+torch.pow((x[1]-mu[1])/sigma[1],2)-2*rho*(x[0]-mu[0])*(x[1]-mu[1])/(sigma[0]*sigma[1])
    return 1/(2*np.pi*sigma[0]*sigma[1]*torch.sqrt(1-rho*rho))*torch.exp(-z/(2*(1-rho*rho)))

def train(rnn : GeneratorRNN, strokes):
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
    strokes_var = Variable(torch.from_numpy(strokes).float(), requires_grad=False)
    hidden = rnn.init_hidden()
    loss = 0
    for i in range(len(strokes)):
        x = strokes_var[i].view(1,-1)
        e,pi,mu,sigma,rho,hidden = rnn(x, hidden)
        y = (e,pi,mu,sigma,rho)
        loss += -torch.log(prob(x,y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_all(rnn : GeneratorRNN, data):
    for strokes in tqdm(data):
        train(rnn, strokes)

if __name__=='__main__':
    data = load_data()
    rnn = GeneratorRNN(1)
    train_all(rnn, data)
    strokes = generate_sequence(rnn, 100)
    plot_stroke(strokes, 'output/1.png')
