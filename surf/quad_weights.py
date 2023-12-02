"""
Created on Wed Dec  1 08:22:31 2023

Copyright 2023 by Gentian Zavalani.
"""
# Standard library imports:
import numpy as np

def quadwts(n):
    '''Quadrature weights for Chebyshev points of 2nd kind.
   quadwts(n) returns the n weights for Clenshaw-Curtis quadrature on 2nd-kind
   Chebyshev points. '''
    c = 2/np.concatenate(([1],  1 - np.arange(2, n, 2)**2), axis=0)
    c = np.concatenate((c, c[int(n/2)-1:0:-1]), axis=0)
    w = np.real(np.fft.ifft(c))
    w[0] = w[0]/2
    w = np.concatenate((w, [w[0]]), axis=0)
    return w
