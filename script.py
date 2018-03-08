import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from utils import plot_stroke

with open('data/strokes.npy','rb') as f:
    strokes = np.load(f,encoding='bytes')

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
        output, (hidden_state, cell_state) = self.lstm(input, hidden)
        output = self.linear(output)
        output = output.view(-1)
        e,pi,mu,sigma,rho = self.split_outputs(output)
        e = 1/(1+torch.exp(e))
        pi = self.softmax(pi)
        mu = mu
        sigma = torch.exp(sigma)
        rho = torch.tanh(rho)
        return e,pi,mu,sigma,rho,hidden_state,cell_state

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
        e,pi,mu,sigma,rho,hidden_state,cell_state = rnn.forward(inputs, hidden)

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
        strokes[i] = [lift,strokes[i-1][1]+sample[0],strokes[i-1][2]+sample[1]]

    return strokes
