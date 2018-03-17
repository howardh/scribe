import torch
import torch.nn.functional as F
from torch.autograd import Variable

class SplitBGMInputFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, input, bias=0):
        n = int((input.size()[2]-1)/6)

        e = input[:,:,0]

        start = 1
        end = 1+n
        pi = input[:,:,start:end]

        start = end
        end = end+2*n
        mu = input[:,:,start:end]

        start = end
        end = end+2*n
        sigma = input[:,:,start:end]

        start = end
        end = end+n
        rho = input[:,:,start:end]

        return e,pi,mu,sigma,rho

    @staticmethod
    def backward(self, grad_e, grad_pi, grad_mu, grad_sigma, grad_rho):
        """
        grad_output -- gradient of loss wrt inputs
        """
        grad_e = grad_e.view(grad_e.size()[0],grad_e.size()[1],1)
        temp = [grad_e,grad_pi,grad_mu,grad_sigma,grad_rho]
        return torch.cat(temp,2)

class BivariateGaussianMixtureLayer(torch.nn.Module):
    def __init__(self, num_components=1):
        super(BivariateGaussianMixtureLayer,self).__init__()
        # See page 23 for parameters
        self.num_components = num_components
        self.split = SplitBGMInputFunction.apply
        self.softmax = torch.nn.Softmax(dim=2)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, bias=0):
        e,pi,mu,sigma,rho = self.split(input)

        n = self.num_components
        seq_len = input.size()[0]
        e = self.sigmoid(-e)
        pi = self.softmax(pi*(1+bias))
        mu = mu.contiguous().view(seq_len,-1,n,2)
        sigma = torch.exp(sigma-bias).view(seq_len,-1,n,2)
        rho = self.tanh(rho)
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
        input -- 2D array
            batch size * features
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
        seq_len = sequence.size()[1]
        if self.is_cuda():
            r = Variable(torch.arange(seq_len).view(seq_len,1).cuda(), requires_grad=False)
        else:
            r = Variable(torch.arange(seq_len).view(seq_len,1), requires_grad=False)
        window_weight = a*torch.exp(-b*torch.pow(k-r,2))
        window_weight = torch.sum(window_weight,dim=2)
        window_weight = window_weight.view(-1,1,seq_len)
        window = window_weight.bmm(sequence) # [-1,seq_len] * [-1,seq_len, features]
        window = window.permute(1,0,2)

        terminal = WindowLayer.is_terminal(k, seq_len)

        return window, hidden, terminal

    def init_hidden(self, batch_size=1):
        if self.is_cuda():
            return Variable(torch.zeros(batch_size,self.num_components).cuda())
        else:
            return Variable(torch.zeros(batch_size,self.num_components))

    @staticmethod
    def is_terminal(k, seq_len):
        # This only works for single sequences. Doesn't apply to batch.
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

    def init_hidden(self, batch_size=1):
        if self.is_cuda():
            hidden1 = Variable(torch.zeros(1, batch_size, 400).cuda())
            cell1 = Variable(torch.zeros(1, batch_size, 400).cuda())
            outputw = Variable(torch.zeros(batch_size, self.num_chars).cuda())
            hidden2 = Variable(torch.zeros(1, batch_size, 400).cuda())
            cell2 = Variable(torch.zeros(1, batch_size, 400).cuda())
        else:
            hidden1 = Variable(torch.zeros(1, batch_size, 400))
            cell1 = Variable(torch.zeros(1, batch_size, 400))
            outputw = Variable(torch.zeros(batch_size, self.num_chars))
            hidden2 = Variable(torch.zeros(1, batch_size, 400))
            cell2 = Variable(torch.zeros(1, batch_size, 400))
        hiddenw = self.window.init_hidden(batch_size)
        return (hidden1, cell1), hiddenw, outputw, (hidden2, cell2)

    def is_cuda(self):
        return next(self.parameters()).is_cuda

