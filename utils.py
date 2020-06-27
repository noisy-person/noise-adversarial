import torch 
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

PAD_IDX = 1


class idx2emb(nn.Module):
    def __init__(self, FLAGS):
        super(idx2emb, self).__init__()
        self.FLAGS = FLAGS
        
        V = FLAGS.embed_num
        D = FLAGS.embed_dim

        with open(FLAGS.emb_path,'rb') as f:
            pretrained=pickle.load(f)
        self.embed = nn.Embedding(V, D,padding_idx=PAD_IDX)
        self.embed.weight.data.copy_(torch.FloatTensor(pretrained))

    def forward(self, x):

        return self.embed(x)


def sequence_mask(length, maxlen=None):
    if maxlen is None:
        maxlen = length.max()
    row_vector = torch.arange(0, maxlen, 1).to('cpu')
    matrix = torch.unsqueeze(length, dim=-1)
    mask = row_vector < matrix

    mask = mask.float()
    return mask

def mask_by_length(feature, length):
    mask = sequence_mask(length) #(B,T)
    mask = torch.unsqueeze(mask,dim=-1) #(B,T,1)
    return feature * mask 

def get_length(feature):
    seq_length = feature.size()[1]
    one_sum = torch.sum( \
        torch.eq( \
            feature,torch.ones(feature.size(),dtype=torch.long)).long() \
                ,-1)

    return seq_length-one_sum





def kl_div_with_logit(p_logit, q_logit):

    p = F.softmax(p_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)

    plogp = ( p *logp).sum(dim=1).mean(dim=0)
    plogq = ( p *logq).sum(dim=1).mean(dim=0)

    return plogp - plogq
"""
def _l2_normalize(d,norm_length):

    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)
"""
def _l2_normalize(d,norm_length):
    # shape(x) = (batch, num_timesteps, d)
    # Divide x by max(abs(x)) for a numerically stable L2 norm.
    # 2norm(x) = a * 2norm(x/a)
    # Scale over the full sequence, dims (1, 2)  
    alpha = torch.max(torch.abs(d),-1,keepdim=True).values + 1e-24
    #  l2_norm = alpha * tf.sqrt(
    #  tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
    l2_norm = alpha * torch.sqrt(
        torch.sum((d/alpha)**2, (1,2),keepdim=True)  + 1e-16
    )
    d_unit = d/ l2_norm
    return norm_length * d_unit


def vat_loss(model, feature, logit, input_length, xi=1e-6, eps=2.5, num_iters=1):

    # find r 

    d = torch.Tensor(feature.size()).normal_()
    for i in range(num_iters):
        #d = 1e-3 *_l2_normalize(mask_by_length(d,input_length))
        d = _l2_normalize(
            mask_by_length(d,input_length) , xi)
        d = Variable(d.cuda(), requires_grad=True)
        y_hat = model(feature.detach() + d)
        delta_kl = kl_div_with_logit(logit.detach(), y_hat)
        delta_kl.backward()

        d = d.grad.data.clone().cpu()
        model.zero_grad()

    #d = _l2_normalize(d)
    d=_l2_normalize(d,eps)
    d = Variable(d.cuda())
    #r_adv = eps *d
    # compute lds
    y_hat = model(feature + d.detach())
    #y_hat = model(feature + r_adv.detach())
    delta_kl = kl_div_with_logit(logit.detach(), y_hat)
    return delta_kl


def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)