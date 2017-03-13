#!/usr/bin/python3

import pandas as pd
import numpy as np
import scipy.sparse as sparse

def baseline_ALS(y, lam, p, a=0, k=75, nIter=10):
    '''
    Function for Baseline Correction with  Asymmetric Least Squares Smoothing (ALS).
    For series of peaks that are either all positive or all negative *).
    
    
    Parameters 
    ----------
    Parameters have to be tuned to the data by hand
    
    y : numpy 1darray 
    p : Asymmetry. *) found that generally 0.001 ≤ p ≤ 0.1 is a good choice (but exceptions may occur)
    lam :  Smoothness penalty factor λ. *) found 10**2 ≤ λ ≤ 10**9 (exceptions may occur)
    nIter : Number of iterations for ALS
     
    *)Reference: Paul H. C. Eilers, Hans F.M. Boelens, Leiden University Medical Centre, October 21, 2005
    https://zanran_storage.s3.amazonaws.com/www.science.uva.nl/ContentPages/443199618.pdf
    
    Additional penalty on non-linear behaviour
    ------------------------------------------
    Experimental.
    
    a : Weight for penalty on path length (trajectory)
    k : Number of neighours to be konsidered
    '''
    
    l = len(y)
    
    # Initialize weights
    w = np.ones(l) # Least squares
    alpha = np.ones(0) # Non-linear behaviour
    
    # Differance matrix for Smoothing
    D = sparse.csc_matrix(np.diff(np.eye(l), 2))
    
    
    # Initialize matrix for penalty on non-linear behaviour
    # ----------------------------------------------------
    nl = np.ones((k*2,l))
    nlDiags = [i for i in range(-k,k+1) if i != 0]

    nlNeighbours = sparse.spdiags(nl, nlDiags, l, l)

    # Substract z_{i} -2k times
    d1 = -2*k * np.ones(l)
    z_i = sparse.spdiags(d1, [0], l, l)

    # Consider values at borders of the array
    d2 = [(k-i) for i in range(k)]
    x = np.zeros(l)
    x[0:len(d2)] += np.array(d2)
    x[-len(d2):][::-1] += np.array(d2)
    nlB = sparse.spdiags(x, [0], l, l)

    K = z_i + nlB + nlNeighbours
    # ----------------------------------------------------
        
    
    # ALS
    for i in range(nIter):
        W = sparse.spdiags(w, 0, l, l)
        A = sparse.spdiags(alpha, 0, l, l)
        Z = W + lam * D.dot(D.transpose()) - A*K # -A*K only for non-linear penalty
        z = sparse.linalg.spsolve(Z, W*y)
        w = p * (y > z) + (1-p) * (y < z)
    
    return z


def baseline_LSNF(y, lam1=6e6, lam2=4e5, p1=5, p2=10, dy=0.01, m=30, a=0, k=75,
                  boundaries=True):
    '''
    Function for Baseline Correction using Least Squares Smoothing with Neighbour
    Depending Freezing (LSNF). For series of peaks which are positive and negative.
    Data should contain slices without peaks.
    
    
    Parameters 
    ----------
    Parameters have to be tuned to the data by hand
    
    y : numpy 1darray 
    boundaries : 'True' for assuming that the boundaries are on the baseline
    p0 : Weight for first least squares fit
    p :
    lam : Smoothness penalty factor λ
    dy : intervall for freezing
    m : Number of neighours in intervall to be needed for freezing
    
    
    Additional penalty on non-linear behaviour
    ------------------------------------------
    Experimental.
    
    a : Weight for penalty on path length (trajectory)
    k : Number of neighours to be konsidered
    
    
    Returns z, freeze
    -----------------
    z : The baseline
    freeze : boolean array
    '''
    

    l = len(y)
    
    # Differance matrix for Smoothing
    D = sparse.csc_matrix(np.diff(np.eye(l), 2))
    
    
    # Initialize matrix for penalty on non-linear behaviour
    # ----------------------------------------------------
    nl = np.ones((k*2,l))
    nlDiags = [i for i in range(-k,k+1) if i != 0]

    nlNeighbours = sparse.spdiags(nl, nlDiags, l, l)

    # Substract z_{i} -2k times
    d1 = -2*k * np.ones(l)
    z_i = sparse.spdiags(d1, [0], l, l)

    # Consider values at borders of the array
    d2 = [(k-i) for i in range(k)]
    x = np.zeros(l)
    x[0:len(d2)] += np.array(d2)
    x[-len(d2):][::-1] += np.array(d2)
    nlB = sparse.spdiags(x, [0], l, l)

    K = z_i + nlB + nlNeighbours
    # ----------------------------------------------------
        
    
    
    
    # LSNF
    # Initialize matrix F for neighbour depending freezing
    # ----------------------------------------------------
    f = np.ones((m*2+1,l))
    fDiags = [i for i in range(-m,m+1)]
    fNeighbours = sparse.spdiags(f, fDiags, l, l)

    # Consider values at borders of the array
    d2 = [(m-i) for i in range(m)]
    x = np.zeros(l)
    x[0:len(d2)] += np.array(d2)
    x[-len(d2):][::-1] += np.array(d2)
    fB = sparse.spdiags(x, [0], l, l)

    F = fB + fNeighbours
    # ----------------------------------------------------



    # LSNF Step one: find 'rough' parts in data
    w = np.ones(l) * p1 # Least squares
    W = sparse.spdiags(w, 0, l, l)
    Z = W + lam1 * D.dot(D.transpose())
    z = sparse.linalg.spsolve(Z, W*y)


    # LSNF Step two: smoothen only rough parts with other areas of z frozen
    alpha = np.ones(0) # Weight non-linear behaviour

    
    # A value is only frozen when m neighbours on each side are in
    # the interval: y-dy <= z <= y+dy
    freeze = F * ((y-dy <= z) & (z <= y+dy)) == (2*m+1) # (True + True + False) = 2
    notFreeze = F * ((y-dy <= z) & (z <= y+dy)) != (2*m+1)

    # Fix borders
    if boundaries:
        freeze[:m] = True; freeze[-m:] = True
        notFreeze[:m] = False; notFreeze[-m:] = False

    w = p2 * freeze + 0 * notFreeze
    alpha = 0 * freeze + a * notFreeze

    W = sparse.spdiags(w, 0, l, l)
    A = sparse.spdiags(alpha, 0, l, l)
    Z = W + lam2 * D.dot(D.transpose()) - A*K # -A*K only for non-linear penalty
    z = sparse.linalg.spsolve(Z, W*y)
    
    return z, freeze
