"""Semantic TCN: no parent/latent input, no right temporal padding."""
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
sys.dont_write_bytecode=True
import torch
from torch import nn
from torch.nn import functional as F
class CausalConv(nn.Module):
    def __init__(self,inc,out,dilation):
        super().__init__();self.left=2*dilation;self.conv=nn.Conv1d(inc,out,3,dilation=dilation)
    def forward(self,x):return self.conv(F.pad(x,(self.left,0)))
class SemanticTCN(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(CausalConv(36,32,1),nn.SiLU(),CausalConv(32,32,2),nn.SiLU(),CausalConv(32,32,4),nn.SiLU())
        self.queue=nn.Linear(32,6);self.switch=nn.Linear(32,7)
    def forward(self,x):
        assert x.ndim==4 and tuple(x.shape[1:])==(4,9,13)
        h=self.encoder(x.reshape(len(x),36,13))[:,:,-1];return self.queue(h),self.switch(h)
    def semantics(self,x):
        q,s=self(x);p=torch.sigmoid(q);tau=(torch.softmax(s,1)*torch.arange(1,8,device=s.device,dtype=s.dtype)).sum(1,keepdim=True)
        return torch.cat([p,tau],1)
