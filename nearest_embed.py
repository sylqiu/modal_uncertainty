import torch
from torch import nn
from torch.nn import functional as F
from utils import Normalize
from torch.autograd import Function, Variable
import numpy as np
import operator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QuantizeEMA(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, mu=0, sigma=1):
        super().__init__()

        self.dim = dim
        self.init_decay = decay
        self.n_embed = n_embed
        self.decay = torch.ones([n_embed]).to(device) * decay
        self.eps = eps
        self.l2norm = Normalize()

        embed = torch.randn(dim, n_embed) * sigma + mu
        # embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.ones(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input, training=True):
        input = input.permute(0, 2, 3, 1)
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if training:
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot.sum(0).mul_(1 - self.decay)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(embed_sum.mul_(1 - self.decay))
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            # embed_normalized = self.l2norm(self.embed_avg / cluster_size.unsqueeze(0), dim=0)
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = quantize.detach() - input
        quantize = input + (quantize - input).detach()
        quantize = quantize.permute(0, 3, 1, 2)
        # embed_ind = embed_ind.unsqueeze(1)
        return quantize, diff, embed_ind



    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def _update_dict(self, x):
        self.embed.data.copy_(x.transpose(0,1))

    def _compute_weight(self, hist):
        
        M = hist[max(hist.items(), key=operator.itemgetter(1))[0]]
        self.decay = torch.ones([self.n_embed]).to(device) * self.init_decay
        for key, val in hist.items():
            self.decay[int(key)] = 0.99 ** np.maximum(0.06* int(M / (val+1e-2)), 1)
        
        print('decay max {:.4f}, decay min {:.4f}'.format(self.decay.max().item(), self.decay.min().item()))



def initial_quantization(memory, num_instance):
    perm = torch.randperm(memory.shape[0])
    ind = perm[:num_instance]
    return F.embedding(ind.to(device), memory)


class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        # argmin = argmin.view(*input.shape[:-1])
        shifted_shape = [input.shape[0], *list(input.shape[2:]) ,input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) == latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)


class Quantize(nn.Module):
    def __init__(self, embeddings_dim, num_embeddings):
        super(Quantize, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        quantize, emd_ind = nearest_embed(x, self.weight.detach() if weight_sg else self.weight)
        l2_diff = (quantize.detach() - x).pow(2).mean()

        return quantize, l2_diff, emd_ind
    


if __name__ == '__main__':
    q_layer = QuantizeEMA(100, 200)
    # criterion = nn.CrossEntropyLoss(reduction='none')
    # logit = torch.randn(4, 200, 2, 2)
    x = torch.randn(4, 100, 2, 2)
    hist = {'10' : 3000, '2':30}
    q_layer._compute_weight(hist)
    print(q_layer.decay)
    qx, diff, ind = q_layer.forward(x)
    
    # print(qx)
    print(diff)
    print(ind.shape)
    # print( 'quantization norm \n{}'.format(q_layer.embed.norm(2, dim=0).cpu().numpy()))
    # loss = criterion(logit, target=ind)
    # print(loss.shape)
    # print(qx.shape)
    # print(diff.shape)
    # print(ind.shape)
