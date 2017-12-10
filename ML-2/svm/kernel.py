import numpy as np


class Kernel(object):
    """Check kernels here https://en.wikipedia.org/wiki/Support_vector_machine"""
    @staticmethod
    def linear():
        return lambda x, y: np.inner(x, y)

    @staticmethod
    def gaussian(sigma):
        gamma = 1 / (2 * sigma**2)
        K = np.exp(-gamma * ((x - y)**2) )
        return K