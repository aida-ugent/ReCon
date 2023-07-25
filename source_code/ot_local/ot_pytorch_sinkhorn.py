#!/usr/bin/env python
#
# OPTIMAL TRANSPORT NODE
# Implementation of differentiable optimal transport using implicit differentiation. Makes use of Sinkhorn normalization
# to solve the entropy regularized problem (Cuturi, NeurIPS 2013) in the forward pass. The problem can be written as
# Let us write the entropy regularized optimal transport problem in the following form,
#
#    minimize (over P) <P, M> + 1/\gamma KL(P || rc^T)
#    subject to        P1 = r and P^T1 = c
#
# where r and c are m- and n-dimensional positive vectors, respectively, each summing to one. Here m-by-n matrix M is
# the input and m-by-n dimensional positive matrix P is the output. The above problem leads to a solution of the form
#
#   P_{ij} = \alpha_i \beta_j e^{-\gamma M_{ij}}
#
# where \alpha and \beta are found by iteratively applying row and column normalizations.
#
# We also provide an option to parametrize the input in log-space as M_{ij} = -\log Q_{ij} where Q is a positive matrix.
# The matrix Q becomes the input. This is more numerically stable for inputs M with large positive or negative values.
#
# See accompanying Jupyter Notebook at https://deepdeclarativenetworks.com.
#
# Stephen Gould <stephen.gould@anu.edu.au>
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Fred Zhang <frederic.zhang@anu.edu.au>
#

import torch


def sinkhorn(M, r=None, c=None, gamma=10.0, eps=1.0e-6, maxiters=2, logspace=False):
    """
    PyTorch function for entropy regularized optimal transport. Assumes batched inputs as follows:
        M:  (B,H,W) tensor
        r:  (B,H) tensor, (1,H) tensor or None for constant uniform vector 1/H
        c:  (B,W) tensor, (1,W) tensor or None for constant uniform vector 1/W

    You can back propagate through this function in O(TBWH) time where T is the number of iterations taken to converge.
    """

    B, H, W = M.shape
    assert r is None or r.shape == (B, H) or r.shape == (1, H)
    assert c is None or c.shape == (B, W) or c.shape == (1, W)
    assert not logspace or torch.all(M > 0.0)

    r = 1.0 / H if r is None else r.unsqueeze(dim=2)
    c = 1.0 / W if c is None else c.unsqueeze(dim=1)

    if logspace:
        P = torch.pow(M, gamma)
    else:
        P = torch.exp(-1.0 * gamma * (M - torch.amin(M, 2, keepdim=True)))

    for i in range(maxiters):
        alpha = torch.sum(P, 2)
        # Perform division first for numerical stability
        P = P / alpha.view(B, H, 1) * r

        beta = torch.sum(P, 1)
        if torch.max(torch.abs(beta - c)) <= eps:
            break
        P = P / beta.view(B, 1, W) * c

    return P

