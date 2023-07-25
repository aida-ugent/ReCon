import torch
from ot_local.ot_pytorch_sinkhorn import sinkhorn


class OTPlugin:
    def __init__(self, user_num, item_num, sinkhorn_gamma, sinkhorn_maxiter, iterative_optimization=False):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.sinkhorn_gamma = sinkhorn_gamma
        self.sinkhorn_maxiter = sinkhorn_maxiter
        self.iterative_optimization = iterative_optimization
        
    def get_sinkhorn_loss(self, predictions, shape=None):
        if shape is None:
            shape = (self.user_num, self.item_num)

        P = predictions.reshape(shape)
        M = torch.log((1-P)/P)
        

        a = sinkhorn(M.unsqueeze(0), gamma=self.sinkhorn_gamma, maxiters=self.sinkhorn_maxiter)
        sinkhorn_loss = torch.dot(a.flatten(),M.unsqueeze(0).flatten())        
        D = -torch.log(1-predictions)
        sinkhorn_loss += D.sum()
        
        return sinkhorn_loss

