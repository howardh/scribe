import numpy as np
import torch
from torch.autograd import Variable

from script import relative_to_absolute
from script import absolute_to_relative

pairs = [[
            np.array([[0,0,0],[0,1,1],[0,1,1]]),
            np.array([[0,0,0],[0,1,1],[0,2,2]])
        ],[
            np.array([[1,0,0],[0,1,1],[1,1,1]]),
            np.array([[1,0,0],[0,1,1],[1,2,2]])
        ]]

def check_np_equals(a,b):
    s = np.sum(np.abs(a-b))
    assert s == 0, "%s != %s" % (a,b)

def test_relative_to_absolute():
    for rstrokes,astrokes in pairs:
        output = relative_to_absolute(rstrokes)
        yield check_np_equals,output,astrokes

def test_absolute_to_relative():
    for rstrokes,astrokes in pairs:
        output = absolute_to_relative(astrokes)
        yield check_np_equals,output,rstrokes

