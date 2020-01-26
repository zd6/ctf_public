import numpy as np
import collections

def state_centering(state, centers, radius):
    """ Translating state to specified coordinates

    Given the state [dx, dy, channel] and n origins,
    method returns [n, range*2-1, range*2-1, channel] size
    translated spaces.

    * State must be padded previously

    Parameters
    ----------

    state : [numpy.ndarray] 
        The size of [dx, dy, channel] array representing the state.
    centers : [list] 
        Origins coordinates in integer tuples (x,y)
    radius : int
        Range of view.
        Must be an odd number

    Returns
    -------
    It returns centered states of numpy.ndarray.
    """

    assert radius % 2, "Centering range must be an odd number"

    dx, dy, ch = state.shape
    states = []
    for x, y in centers:
        states.append(state[x-radius:x+radius+1, y-radius:y+radius+1, :])

    return np.stack(states)

class Stacked_state:
    def __init__(self, keep_frame, axis):
        self.keep_frame = keep_frame
        self.axis = axis
        self.stack = []

    def initiate(self, obj):
        self.stack = [obj] * self.keep_frame
    
    def __call__(self, obj=None):
        if obj is None:
            return np.concatenate(self.stack, axis=self.axis)
        self.stack.append(obj)
        self.stack.pop(0)
        return np.concatenate(self.stack, axis=self.axis)


class VStacked_state(Stacked_state):
    def __call__(self, obj=None):
        if obj is None:
            return np.stack(self.stack, axis=self.axis)
        self.stack.append(obj)
        self.stack.pop(0)
        return np.stack(self.stack, axis=self.axis)

