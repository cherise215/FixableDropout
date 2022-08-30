import torch.nn as nn
import torch
from torch.nn import functional as F


class FixableDropout1d(nn.Module):
    """
     based on 1D pytorch dropout, supporting lazy load with last generated mask.
     To use last generated mask, set lazy_load to True
    """
    def __init__(self, p: float = 0.5,inplace=False,lazy_load: bool = False,training=True):
        super(FixableDropout1d, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.seed  = None
        self.lazy_load = lazy_load
        self.training=training

    def forward(self, X):
        if self.training:
            if self.lazy_load:
                if not self.seed is None:
                    seed  = self.seed
                else:
                    seed = torch.seed()
            else:seed = torch.seed()
        else:
            seed = torch.seed()
        self.seed=seed
        torch.manual_seed(seed)
        X = F.dropout1d(X, p=self.p, training=self.training, inplace=self.inplace)
        return X
    def fix(self):
        self.lazy_load=True
    def activate(self):
        self.lazy_load=False


class FixableDropout2d(nn.Module):
    """
     based on 2D pytorch dropout, supporting lazy load with last generated mask.
     To use last generated mask, set lazy_load to True
    """
    def __init__(self, p: float = 0.5,inplace=False,lazy_load: bool = False,training=True):
        super(FixableDropout2d, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.seed  = None
        self.lazy_load = lazy_load
        self.training=training

    def forward(self, X):
        if self.training:
            if self.lazy_load:
                if not self.seed is None:
                    seed  = self.seed
                else:
                    seed = torch.seed()
            else:seed = torch.seed()
        else:
            seed = torch.seed()
        self.seed=seed
        torch.manual_seed(seed)
        X = F.dropout2d(X, p=self.p, training=self.training, inplace=self.inplace)
        return X
    def fix(self):
        self.lazy_load=True
    def activate(self):
        self.lazy_load=False

class FixableDropout3d(nn.Module):
    """
     based on 3D pytorch dropout, supporting lazy load with last generated mask
    """
    def __init__(self, p: float = 0.5,inplace=False,lazy_load: bool = False,training=True):
        super(FixableDropout3d, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.seed  = None
        self.lazy_load = lazy_load
        self.training=training

    def forward(self, X):
        if self.training:
            if self.lazy_load:
                if not self.seed is None:
                    seed  = self.seed
                else:
                    seed = torch.seed()
            else:seed = torch.seed()
        else:
            seed = torch.seed()
        self.seed=seed
        torch.manual_seed(seed)
        X = F.dropout3d(X, p=self.p, training=self.training, inplace=self.inplace)
        return X

    def fix(self):
        self.lazy_load=True
    def activate(self):
        self.lazy_load=False

if __name__ =="__main__":
    drop= FixableDropout2d(p=0.5, inplace=False)
    a = torch.ones(1,4,1,1)
    a_1= drop(a)
    print ('first time')
    print (a_1)

    drop.lazy_load=True
    a_2= drop(a)

    print ('reuse the previous dropout mask')
    print (a_2)

    assert (a_1==a_2).all()

    print ('new dropout mask')
    drop.lazy_load=False
    print(drop(a))
