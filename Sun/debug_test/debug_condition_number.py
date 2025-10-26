#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:08:25 2023
@author: F. Neri, TU Delft

check dependency of condition number for some matrices
"""
import numpy as np

#original matrix
M = np.array(([[1, 2, 3],
               [0, 1, 2],
               [0, 0, 1]]))
u, s, v = np.linalg.svd(M)
chi = np.min(s)/np.max(s)

#bc condition subsituted to the first
M = np.array(([[0, 1, 1],
               [0, 1, 2],
               [0, 0, 1]]))
u, s, v = np.linalg.svd(M)
chi1 = np.min(s)/np.max(s)

#bc condition subsituted to the second
M = np.array(([[1, 2, 3],
               [0, 1, 1],
               [0, 0, 1]]))
u, s, v = np.linalg.svd(M)
chi2 = np.min(s)/np.max(s)

#bc condition subsituted to the third
M = np.array(([[1, 2, 3],
               [0, 1, 2],
               [0, 1, 1]]))
u, s, v = np.linalg.svd(M)
chi3 = np.min(s)/np.max(s)

#coherent bc condition appended
M = np.array(([[1, 2, 3],
               [0, 1, 2],
               [0, 0, 1],
               [0, 1, 1]]))
u, s, v = np.linalg.svd(M)
chi4 = np.min(s)/np.max(s)

#incoherent bc condition appended
M = np.array(([[1, 2, 3],
               [0, 1, 2],
               [0, 0, 1],
               [0, 1, -1]]))
u, s, v = np.linalg.svd(M)
chi5 = np.min(s)/np.max(s)


