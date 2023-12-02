"""
Created on Wed Dec  1 08:22:31 2023

Copyright 2023 by Gentian Zavalani.
"""
# Standard library imports:
import numpy as np
from dmsuite.poly_diff import Chebyshev

class SurfceMesh:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.xu = []
        self.yu = []
        self.zu = []
        self.xv = []
        self.yv = []
        self.zv = []
        self.ux = []
        self.uy = []
        self.uz = []
        self.vx = []
        self.vy = []
        self.vz = []
        self.E = []
        self.F = []
        self.G = []
        self.J = []
        self.singular = []
        self.facenormals = []
        self.edgenormals = []
        self.connectivity = []
        nelem = len(x)
        n = len(x[0])
        cheb = Chebyshev(degree=n-1)
        D = cheb.at_order(1)

        for k in range(nelem):
            self.xu.append(np.dot(x[k], D.T))
            self.xv.append(np.dot(D, x[k]))
            self.yu.append(np.dot(y[k], D.T))
            self.yv.append(np.dot(D, y[k]))
            self.zu.append(np.dot(z[k], D.T))
            self.zv.append(np.dot(D, z[k]))

            self.E.append(self.xu[k] * self.xu[k] + self.yu[k] * self.yu[k] + self.zu[k] * self.zu[k])
            self.G.append(self.xv[k] * self.xv[k] + self.yv[k] * self.yv[k] + self.zv[k] * self.zv[k])
            self.F.append(self.xu[k] * self.xv[k] + self.yu[k] * self.yv[k] + self.zu[k] * self.zv[k])
            self.J.append(self.E[k] * self.G[k] - self.F[k] ** 2)

            scl = np.max(np.abs(self.G[k] * self.xu[k] - self.F[k] * self.xv[k]))
            if np.any(np.abs(self.J[k]) < 1e-10 * scl):
                self.singular.append(True)
                self.ux.append(self.G[k] * self.xu[k] - self.F[k] * self.xv[k])
                self.uy.append(self.G[k] * self.yu[k] - self.F[k] * self.yv[k])
                self.uz.append(self.G[k] * self.zu[k] - self.F[k] * self.zv[k])
                self.vx.append(self.E[k] * self.xv[k] - self.F[k] * self.xu[k])
                self.vy.append(self.E[k] * self.yv[k] - self.F[k] * self.yu[k])
                self.vz.append(self.E[k] * self.zv[k] - self.F[k] * self.zu[k])
            else:
                self.ux.append((self.G[k] * self.xu[k] - self.F[k] * self.xv[k]) / self.J[k])
                self.uy.append((self.G[k] * self.yu[k] - self.F[k] * self.yv[k]) / self.J[k])
                self.uz.append((self.G[k] * self.zu[k] - self.F[k] * self.zv[k]) / self.J[k])
                self.vx.append((self.E[k] * self.xv[k] - self.F[k] * self.xu[k]) / self.J[k])
                self.vy.append((self.E[k] * self.yv[k] - self.F[k] * self.yu[k]) / self.J[k])
                self.vz.append((self.E[k] * self.zv[k] - self.F[k] * self.zu[k]) / self.J[k])
