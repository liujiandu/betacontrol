import numpy as np

def ensure_rng(random_state=None):
    """
    Create a random number generator based on an optional seed
    random_state can be an interger, or another random state, 
    or None for an unseded rng
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state

