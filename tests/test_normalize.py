import numpy as np

from data import normalize_strokes
from data import unnormalize_strokes

def test_normalize():
    strokes = [np.array([
        [0,0,0],
        [0,1,1],
        [0,2,3]])]
    output,m,s = normalize_strokes(strokes)
    assert m[0]==1
    assert (m[1]-4/3)<0.00001
    assert (s[0]-0.816496580927726)<0.00001, ("%s != %s" % (s[0],0.816496580927726))
    assert (s[1]-1.5275252316519)<0.00001

    strokes = [np.array([[0,0,0]]),
            np.array([[0,1,1]]),
            np.array([[0,2,3]])]
    output,m,s = normalize_strokes(strokes)
    assert m[0]==1
    assert (m[1]-4/3)<0.00001
    assert (s[0]-0.816496580927726)<0.00001, ("%s != %s" % (s[0],0.816496580927726))
    assert (s[1]-1.5275252316519)<0.00001
